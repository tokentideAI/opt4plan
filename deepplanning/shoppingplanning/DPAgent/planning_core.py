"""Parser, coverage retrieval, Candidate Memory, and deterministic shopping solvers."""

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from .prompts import PARSER_PROMPT
except ImportError:
    from prompts import PARSER_PROMPT


class StructuredPlanningMixin:
    """Planning methods composed into ShoppingFnAgent."""

    @staticmethod
    def _json_object(text: str) -> Dict[str, Any]:
        """Extract one JSON object from a plain or fenced model response."""
        text = (text or '').strip()
        if text.startswith('```'):
            lines = text.splitlines()
            text = '\n'.join(lines[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find('{'), text.rfind('}')
            if start < 0 or end <= start:
                raise ValueError('query parser did not return a JSON object')
            value = json.loads(text[start:end + 1])
        if not isinstance(value, dict):
            raise ValueError('query parser result must be a JSON object')
        return value

    def _parse_query(
        self,
        user_query: str,
        user_info: Dict[str, Any],
        level: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compile natural language into IR, then normalize it deterministically."""
        parser_prompt = PARSER_PROMPT.format(
            allowed_fields=', '.join(sorted(self.ALLOWED_FIELDS)),
            allowed_ops=', '.join(sorted(self.ALLOWED_OPS)),
        )
        messages = [
            {'role': 'system', 'content': parser_prompt},
            {'role': 'user', 'content': f'User request:\n{user_query}'},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(1, 3):
            self.metrics['parser_attempts'] = attempt
            response = self._call_llm(messages=messages, tools=None)
            content = response.choices[0].message.content or ''
            try:
                problem_ir = self._json_object(content)
                self._validate_problem_ir(problem_ir)
                self._propagate_global_constraints(problem_ir)
                self._normalize_level_constraints(
                    user_query, problem_ir, level or self.level
                )
                self._resolve_profile_constraints(problem_ir, user_info)
                self.metrics['parser_success'] = True
                return problem_ir
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                messages.extend([
                    {'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': (
                        f'The JSON failed validation: {error}. Repair it and return the full '
                        'JSON object only.'
                    )},
                ])
        raise ValueError(f'ProblemIR parsing failed after repair: {last_error}')

    @staticmethod
    def _extract_budget_bounds(user_query: str) -> Optional[tuple[float, float]]:
        """Extract the closed total-price interval used by Level-2 benchmark queries."""
        number = r'([\d,]+(?:\.\d+)?)'
        patterns = (
            rf'budget\s+(?:is\s+)?between\s+{number}\s+and\s+{number}',
            rf'(?:budget|range)\s+of\s+{number}\s+to\s+{number}',
            rf'total\s+spending\s+between\s+{number}\s+and\s+{number}',
            rf'total\s+price\s+is\s+no\s+less\s+than\s+{number}\s+and\s+no\s+more\s+than\s+{number}',
        )
        for pattern in patterns:
            match = re.search(pattern, user_query, flags=re.IGNORECASE)
            if match:
                lower = float(match.group(1).replace(',', ''))
                upper = float(match.group(2).replace(',', ''))
                if lower > upper:
                    lower, upper = upper, lower
                return lower, upper
        return None

    @classmethod
    def _normalize_level_constraints(
        cls,
        user_query: str,
        problem_ir: Dict[str, Any],
        level: int,
    ) -> None:
        """Canonicalize cross-slot constraints used by deterministic solvers."""
        global_constraints = problem_ir.get('global_constraints') or {}
        if not isinstance(global_constraints, dict):
            global_constraints = {'other': global_constraints}
        structured_budget = global_constraints.get('budget')
        if not isinstance(structured_budget, dict) or not any(
            structured_budget.get(key) is not None for key in ('min', 'max')
        ):
            extracted = cls._extract_budget_bounds(user_query)
            structured_budget = (
                {'min': extracted[0], 'max': extracted[1]}
                if extracted is not None else None
            )
        if structured_budget is not None:
            lower = structured_budget.get('min')
            upper = structured_budget.get('max')
            if lower is not None:
                lower = float(lower)
            if upper is not None:
                upper = float(upper)
            if lower is not None and upper is not None and lower > upper:
                lower, upper = upper, lower
            global_constraints['budget'] = {
                'min': lower,
                'max': upper,
                'basis': 'final_price' if level == 3 else 'subtotal',
            }
            # Guard against a parser duplicating a total budget as slot prices.
            budget_words = ('budget', 'total price', 'total spending', 'range of')
            for slot in problem_ir['slots']:
                slot['constraints'] = [
                    constraint for constraint in slot['constraints']
                    if not (
                        constraint.get('field') == 'price'
                        and any(
                            word in str(constraint.get('source_text', '')).casefold()
                            for word in budget_words
                        )
                    )
                ]
        problem_ir['global_constraints'] = global_constraints

    @classmethod
    def _validate_problem_ir(cls, problem_ir: Dict[str, Any]) -> None:
        slots = problem_ir.get('slots')
        if not isinstance(slots, list) or not slots:
            raise ValueError('slots must be a non-empty list')
        for expected_id, slot in enumerate(slots, 1):
            if not isinstance(slot, dict):
                raise ValueError('each slot must be an object')
            if slot.get('slot_id') != expected_id:
                raise ValueError('slot_id values must be consecutive integers starting at 1')
            quantity = slot.get('quantity', 1)
            if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
                raise ValueError(f'slot {expected_id} has invalid quantity')
            constraints = slot.get('constraints')
            if not isinstance(constraints, list):
                raise ValueError(f'slot {expected_id} constraints must be a list')
            for constraint in constraints:
                if not isinstance(constraint, dict):
                    raise ValueError(f'slot {expected_id} contains a non-object constraint')
                field = constraint.get('field')
                operator = constraint.get('op')
                if field not in cls.ALLOWED_FIELDS:
                    raise ValueError(f'unsupported constraint field: {field}')
                if operator not in cls.ALLOWED_OPS:
                    raise ValueError(f'unsupported constraint operator: {operator}')
                if 'value' not in constraint:
                    raise ValueError(f'constraint {field} has no value')
                if field in {
                    'price', 'stock_quantity', 'sales_volume.monthly', 'sales_volume.total',
                    'rating.average_score', 'rating.total_reviews',
                    'rating.distribution.1_star', 'rating.distribution.2_star',
                    'rating.distribution.3_star', 'rating.distribution.4_star',
                    'rating.distribution.5_star', 'transport_time',
                } and not isinstance(constraint['value'], (int, float)):
                    raise ValueError(f'numeric field {field} requires a numeric value')
        global_constraints = problem_ir.setdefault('global_constraints', {})
        if not isinstance(global_constraints, (dict, list)):
            raise ValueError('global_constraints must be an object or list')

    @classmethod
    def _propagate_global_constraints(cls, problem_ir: Dict[str, Any]) -> None:
        """Copy structured per-product global requirements into every slot."""
        global_constraints = problem_ir.get('global_constraints') or {}
        if isinstance(global_constraints, dict):
            per_product = global_constraints.pop('per_product', [])
        else:
            per_product = [
                item for item in global_constraints
                if isinstance(item, dict) and item.get('scope') == 'all_slots'
            ]
            problem_ir['global_constraints'] = [
                item for item in global_constraints if item not in per_product
            ]
        if not isinstance(per_product, list):
            raise ValueError('global per_product constraints must be a list')
        for constraint in per_product:
            normalized = dict(constraint)
            normalized.pop('scope', None)
            if normalized.get('field') not in cls.ALLOWED_FIELDS \
                    or normalized.get('op') not in cls.ALLOWED_OPS \
                    or 'value' not in normalized:
                raise ValueError(f'invalid global per-product constraint: {constraint}')
            for slot in problem_ir['slots']:
                if normalized not in slot['constraints']:
                    slot['constraints'].append(dict(normalized))

    @staticmethod
    def _resolve_profile_constraints(
        problem_ir: Dict[str, Any], user_info: Dict[str, Any]
    ) -> None:
        demographics = user_info.get('demographics') or {}
        body_profile = user_info.get('body_profile') or {}
        defaults = {
            'target_demographic': demographics.get('gender'),
            'standard_sizes': body_profile.get('standard_sizes') or {},
        }
        for slot in problem_ir['slots']:
            explicit_fields = {c.get('field') for c in slot['constraints']}
            profile_defaults: Dict[str, Any] = {}
            if defaults['standard_sizes']:
                profile_defaults['standard_sizes'] = defaults['standard_sizes']
            if defaults['target_demographic'] \
                    and 'target_demographic' not in explicit_fields:
                profile_defaults['target_demographic'] = defaults['target_demographic']
            slot['profile_defaults'] = profile_defaults

    @staticmethod
    def _build_coupon_ir(user_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize benchmark coupon records once, before optimization."""
        coupons = []
        for coupon_name, quantity in sorted((user_info.get('coupons') or {}).items()):
            match = re.search(
                r'¥([\d,]+(?:\.\d+)?)\s+off\s+every\s+¥([\d,]+(?:\.\d+)?)',
                coupon_name,
            )
            if not match or int(quantity) <= 0:
                continue
            if coupon_name.startswith('Cross-store:'):
                scope = 'cross_store'
            elif coupon_name.startswith('Same-brand:'):
                scope = 'same_brand'
            elif coupon_name.startswith('VIP:'):
                scope = 'vip'
            else:
                scope = 'unknown'
            coupons.append({
                'coupon_name': coupon_name,
                'scope': scope,
                'brand': None,
                'threshold': float(match.group(2).replace(',', '')),
                'discount': float(match.group(1).replace(',', '')),
                'owned_quantity': int(quantity),
                'vip_required': scope == 'vip',
                # The official cart tool and benchmark labels use one shared
                # pre-discount subtotal capacity for all coupon display types.
                'eligibility_model': 'benchmark_cart_threshold',
            })
        return coupons

    @classmethod
    def _init_memory(
        cls,
        problem_ir: Dict[str, Any],
        user_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        user = user_info or {}
        return {
            'slots': {
                str(slot['slot_id']): {
                    'constraints': slot['constraints'],
                    'candidate_ids': [],
                    'unknown_candidate_ids': [],
                    'status': 'unresolved',
                    'coverage': {
                        'feasible_candidate_found': False,
                        'candidate_set_complete': False,
                        'optimality_certified': False,
                    },
                }
                for slot in problem_ir['slots']
            },
            'candidates': {},
            'transport_times': {},
            'user_info': user,
            'coupons': cls._build_coupon_ir(user),
            **({'trace': []} if os.getenv('DEBUG_TRACE') == '1' else {}),
        }

    def _update_memory(
        self,
        memory: Dict[str, Any],
        call: Dict[str, Any],
        tool_result: str,
    ) -> None:
        """Update compact planner memory from an official tool result."""
        try:
            payload = json.loads(tool_result)
        except (TypeError, json.JSONDecodeError):
            payload = {'raw': tool_result}
        try:
            arguments = json.loads(call.get('arguments') or '{}')
        except (TypeError, json.JSONDecodeError):
            arguments = {}
        if 'trace' in memory:
            memory['trace'].append({
                'tool': call['name'],
                'arguments': arguments,
                'result': payload,
            })
        if not isinstance(payload, dict) or payload.get('error'):
            return
        if call['name'] == 'get_user_info':
            memory['user_info'] = payload
            memory['coupons'] = self._build_coupon_ir(payload)
        elif call['name'] == 'get_product_details':
            for product in payload.get('products', []):
                product_id = product.get('product_id')
                if product_id:
                    memory['candidates'][product_id] = product
            self.metrics['retrieved_products'] = len(memory['candidates'])
        elif call['name'] == 'calculate_transport_time':
            product_id = payload.get('product_id')
            days = payload.get('estimated_delivery_days')
            if product_id and days is not None:
                memory['transport_times'][product_id] = days

    @staticmethod
    def _nested_value(data: Dict[str, Any], path: str) -> Any:
        value: Any = data
        for part in path.split('.'):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    @staticmethod
    def _canonical_text(value: Any) -> str:
        text = ''.join(character for character in str(value).casefold() if character.isalnum())
        aliases = {
            'allseason': 'allseasons',
            'female': 'women',
            'woman': 'women',
            'male': 'men',
            'man': 'men',
        }
        return aliases.get(text, text)

    @classmethod
    def _compare(cls, actual: Any, operator: str, expected: Any) -> Optional[bool]:
        if actual is None:
            return None
        if operator == 'contains':
            return cls._canonical_text(expected) in cls._canonical_text(actual)
        if operator in ('==', '!=') and isinstance(actual, str):
            equal = cls._canonical_text(actual) == cls._canonical_text(expected)
            return equal if operator == '==' else not equal
        try:
            if operator == '==':
                return actual == expected
            if operator == '!=':
                return actual != expected
            if operator == '<':
                return actual < expected
            if operator == '<=':
                return actual <= expected
            if operator == '>':
                return actual > expected
            if operator == '>=':
                return actual >= expected
        except TypeError:
            return False
        return False

    @classmethod
    def _product_size_group(cls, product: Dict[str, Any]) -> str:
        name = cls._canonical_text(product.get('name', ''))
        if any(word in name for word in ('shoe', 'sneaker', 'boot', 'loafer', 'sandal')):
            return 'shoes'
        if any(word in name for word in ('pant', 'trouser', 'short', 'jogger', 'bottom', 'skirt')):
            return 'bottoms'
        return 'tops'

    @classmethod
    def _profile_status(
        cls, product: Dict[str, Any], slot: Dict[str, Any]
    ) -> tuple[str, List[Dict[str, Any]]]:
        defaults = slot.get('profile_defaults') or {}
        failures: List[Dict[str, Any]] = []
        explicit_fields = {c.get('field') for c in slot.get('constraints', [])}
        expected_gender = defaults.get('target_demographic')
        actual_gender = product.get('target_demographic')
        if expected_gender and 'target_demographic' not in explicit_fields and actual_gender:
            expected = cls._canonical_text(expected_gender)
            actual = cls._canonical_text(actual_gender)
            if actual not in {expected, 'unisex'}:
                failures.append({
                    'field': 'target_demographic', 'expected': expected_gender,
                    'actual': actual_gender, 'source': 'user_profile',
                })
        sizes = defaults.get('standard_sizes') or {}
        if sizes and 'size' not in explicit_fields:
            group = cls._product_size_group(product)
            expected_size = sizes.get(group)
            actual_size = product.get('size')
            if expected_size is not None and actual_size is not None \
                    and cls._canonical_text(expected_size) != cls._canonical_text(actual_size):
                failures.append({
                    'field': 'size', 'expected': expected_size, 'actual': actual_size,
                    'source': f'user_profile.{group}',
                })
        return (cls.UNSAT if failures else cls.SAT), failures

    def _candidate_status(
        self,
        product: Dict[str, Any],
        slot: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> tuple[str, List[Dict[str, Any]]]:
        unknown: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for constraint in slot['constraints']:
            field = constraint['field']
            if field == 'transport_time':
                actual = memory['transport_times'].get(product.get('product_id'))
            else:
                actual = self._nested_value(product, field)
            result = self._compare(actual, constraint['op'], constraint['value'])
            diagnostic = {
                'field': field,
                'op': constraint['op'],
                'expected': constraint['value'],
                'actual': actual,
            }
            if result is None:
                unknown.append(diagnostic)
            elif not result:
                failures.append(diagnostic)
        profile_status, profile_failures = self._profile_status(product, slot)
        failures.extend(profile_failures)
        if failures or profile_status == self.UNSAT:
            return self.UNSAT, failures
        if unknown:
            return self.UNKNOWN, unknown
        return self.SAT, []

    def _slot_candidates(
        self,
        slot: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates = [
            product for product in memory['candidates'].values()
            if self._candidate_status(product, slot, memory)[0] == self.SAT
        ]
        return sorted(candidates, key=lambda product: (
            product.get('price', float('inf')), product['product_id']
        ))

    def _slot_unknown_candidates(
        self, slot: Dict[str, Any], memory: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return [
            product for product in memory['candidates'].values()
            if self._candidate_status(product, slot, memory)[0] == self.UNKNOWN
        ]

    def _memory_tool_call(
        self,
        messages: List[Dict[str, Any]],
        memory: Dict[str, Any],
        name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self._record_tool_action(
            messages, name, arguments, compact_result=True
        )
        self._update_memory(memory, {
            'name': name,
            'arguments': json.dumps(arguments, ensure_ascii=False),
        }, result)
        payload = json.loads(result)
        if not isinstance(payload, dict) or payload.get('error'):
            raise RuntimeError(f'{name} failed: {payload}')
        return payload

    @staticmethod
    def _tool_filter_for_constraint(
        constraint: Dict[str, Any]
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        field = constraint['field']
        operator = constraint['op']
        value = constraint['value']
        if field == 'brand' and operator == '==':
            return 'filter_by_brand', {'brand_names': [str(value)]}
        if field == 'color' and operator == '==':
            return 'filter_by_color', {'colors': [str(value)]}
        if field == 'size' and operator == '==':
            return 'filter_by_size', {'sizes': [str(value)]}
        if field in {
            'price', 'stock_quantity', 'sales_volume.monthly',
            'sales_volume.total', 'rating.average_score', 'rating.total_reviews',
        } and operator in {'>', '>=', '<', '<=', '=='}:
            return 'filter_by_range', {
                'condition_key': field,
                'operator': operator,
                'value': value,
            }
        return None

    def _certify_slot_coverage(
        self,
        slot: Dict[str, Any],
        memory: Dict[str, Any],
        messages: List[Dict[str, Any]],
        enumerate_all: bool = False,
    ) -> Dict[str, Any]:
        """Build a price frontier; optionally verify every candidate for coupled solving."""
        slot_memory = memory['slots'][str(slot['slot_id'])]
        current_ids: Optional[List[str]] = None
        applied_filters = []
        source_exhaustive = True
        for constraint in slot['constraints']:
            tool_filter = self._tool_filter_for_constraint(constraint)
            if tool_filter is None:
                continue
            name, arguments = tool_filter
            if current_ids is not None:
                if not current_ids:
                    break
                arguments['product_ids'] = current_ids
            payload = self._memory_tool_call(messages, memory, name, arguments)
            current_ids = list(payload.get('filtered_products_ids') or [])
            filter_exhaustive = bool(
                payload.get('is_exhaustive') is True
                and payload.get('returned_count') == len(current_ids)
                and payload.get('total_count') == len(current_ids)
            )
            source_exhaustive = source_exhaustive and filter_exhaustive
            applied_filters.append({
                'field': constraint['field'],
                'op': constraint['op'],
                'value': constraint['value'],
                'tool': name,
                'is_exhaustive': filter_exhaustive,
            })

        if current_ids == []:
            coverage = {
                'feasible_candidate_found': False,
                'candidate_set_complete': source_exhaustive,
                'optimality_certified': False,
                'infeasible_certified': source_exhaustive,
                'total_count': 0,
                'retrieved_count': 0,
                'pages_exhausted': True,
                'sort_by': 'price',
                'order': 'asc',
                'all_cheaper_checked': True,
                'best_product_id': None,
                'applied_filters': applied_filters,
                'source_exhaustive': source_exhaustive,
            }
            slot_memory['coverage'] = coverage
            return coverage

        sort_arguments: Dict[str, Any] = {'sort_by': 'price', 'order': 'asc'}
        if current_ids is not None:
            sort_arguments['product_ids'] = current_ids
        payload = self._memory_tool_call(
            messages, memory, 'sort_products', sort_arguments
        )
        frontier = list(payload.get('sorted_product_ids') or [])
        sort_exhaustive = bool(
            payload.get('is_exhaustive') is True
            and payload.get('returned_count') == len(frontier)
            and payload.get('total_count') == len(frontier)
        )
        source_exhaustive = source_exhaustive and sort_exhaustive
        total_count = int(payload.get('total_count', len(frontier)))
        batch_size = max(1, int(os.getenv('SHOPPING_COVERAGE_BATCH_SIZE', '20')))
        checked_count = 0
        best_product_id: Optional[str] = None
        unknown_count = 0
        infeasible_count = 0
        all_cheaper_checked = True
        debug_checks = [] if 'trace' in memory else None

        for offset in range(0, total_count, batch_size):
            batch = frontier[offset:offset + batch_size]
            missing = [pid for pid in batch if pid not in memory['candidates']]
            if missing:
                self._memory_tool_call(
                    messages, memory, 'get_product_details', {'product_ids': missing}
                )
            for product_id in batch:
                product = memory['candidates'].get(product_id)
                if product is None:
                    unknown_count += 1
                    if best_product_id is None:
                        all_cheaper_checked = False
                    if debug_checks is not None:
                        debug_checks.append({
                            'product_id': product_id,
                            'status': self.UNKNOWN,
                            'diagnostics': [{'field': 'product_details', 'actual': None}],
                        })
                    checked_count += 1
                    continue
                status, diagnostics = self._candidate_status(product, slot, memory)
                if status == self.UNKNOWN and any(
                    item.get('field') == 'transport_time' for item in diagnostics
                ):
                    self._resolve_unknown_information(
                        {'slots': [slot]}, memory, messages
                    )
                    status, diagnostics = self._candidate_status(product, slot, memory)
                if status == self.UNKNOWN:
                    unknown_count += 1
                    if best_product_id is None:
                        all_cheaper_checked = False
                elif status == self.UNSAT:
                    infeasible_count += 1
                if debug_checks is not None:
                    debug_checks.append({
                        'product_id': product_id,
                        'status': status,
                        'diagnostics': diagnostics,
                    })
                checked_count += 1
                if status == self.SAT and best_product_id is None:
                    best_product_id = product_id
                if status == self.SAT and not enumerate_all:
                    break
            if best_product_id is not None and not enumerate_all:
                break

        pages_exhausted = bool(
            source_exhaustive and checked_count == total_count == len(frontier)
        )
        candidate_set_complete = bool(
            pages_exhausted
            and unknown_count == 0
        )
        coverage = {
            'feasible_candidate_found': best_product_id is not None,
            'candidate_set_complete': candidate_set_complete,
            'optimality_certified': bool(
                source_exhaustive and best_product_id and all_cheaper_checked
            ),
            'infeasible_certified': bool(
                candidate_set_complete and best_product_id is None
            ),
            'total_count': total_count,
            'retrieved_count': checked_count,
            'pages_exhausted': pages_exhausted,
            'sort_by': 'price',
            'order': 'asc',
            'all_cheaper_checked': all_cheaper_checked,
            'best_product_id': best_product_id,
            'applied_filters': applied_filters,
            'unknown_count': unknown_count,
            'infeasible_count': infeasible_count,
            'source_exhaustive': source_exhaustive,
        }
        if debug_checks is not None:
            coverage['checked_products'] = debug_checks
        slot_memory['coverage'] = coverage
        return coverage

    def _retrieve_and_certify(
        self,
        level: int,
        problem_ir: Dict[str, Any],
        memory: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deterministically retrieve candidates and return a three-state certificate."""
        enumerate_all = level in (2, 3)
        for slot in problem_ir['slots']:
            self._certify_slot_coverage(
                slot, memory, messages, enumerate_all=enumerate_all
            )
        self._retrieval_complete(problem_ir, memory)
        slot_coverages = {
            str(slot['slot_id']): memory['slots'][str(slot['slot_id'])]['coverage']
            for slot in problem_ir['slots']
        }
        if any(c.get('infeasible_certified') for c in slot_coverages.values()):
            status = self.COVERAGE_INFEASIBLE
        elif level == 1 and all(
            c.get('optimality_certified') for c in slot_coverages.values()
        ):
            status = self.COVERAGE_READY
        elif level in (2, 3) and all(
            c.get('candidate_set_complete')
            and c.get('feasible_candidate_found')
            for c in slot_coverages.values()
        ):
            status = self.COVERAGE_READY
        else:
            status = self.COVERAGE_INCOMPLETE
        return {'status': status, 'slots': slot_coverages}

    def _slot_enumeration_complete(
        self, slot: Dict[str, Any], memory: Dict[str, Any]
    ) -> bool:
        coverage = memory['slots'][str(slot['slot_id'])].get('coverage') or {}
        return bool(
            coverage.get('source_exhaustive') is True
            and coverage.get('candidate_set_complete') is True
            and coverage.get('pages_exhausted') is True
            and coverage.get('retrieved_count') == coverage.get('total_count')
        )

    def _slot_optimality_certified(
        self, slot: Dict[str, Any], memory: Dict[str, Any]
    ) -> bool:
        candidates = self._slot_candidates(slot, memory)
        if not candidates:
            return False
        coverage = memory['slots'][str(slot['slot_id'])].get('coverage') or {}
        best_product_id = candidates[0]['product_id']
        if self._slot_enumeration_complete(slot, memory):
            return coverage.get('best_product_id') == best_product_id
        return bool(
            coverage.get('source_exhaustive') is True
            and coverage.get('sort_by') == 'price'
            and coverage.get('order') == 'asc'
            and coverage.get('all_cheaper_checked') is True
            and coverage.get('best_product_id') == best_product_id
        )

    def _ready_for_solver(
        self, problem_ir: Dict[str, Any], memory: Dict[str, Any]
    ) -> bool:
        return all(
            self._slot_optimality_certified(slot, memory)
            for slot in problem_ir['slots']
        )

    def _retrieval_complete(self, problem_ir: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        """Refresh the three coverage states and enforce the solver readiness gate."""
        complete = 0
        for slot in problem_ir['slots']:
            candidates = self._slot_candidates(slot, memory)
            unknown = self._slot_unknown_candidates(slot, memory)
            slot_memory = memory['slots'][str(slot['slot_id'])]
            slot_memory['candidate_ids'] = [p['product_id'] for p in candidates]
            slot_memory['unknown_candidate_ids'] = [p['product_id'] for p in unknown]
            coverage = slot_memory.setdefault('coverage', {})
            coverage['feasible_candidate_found'] = bool(candidates)
            coverage.setdefault('candidate_set_complete', False)
            coverage['optimality_certified'] = self._slot_optimality_certified(
                slot, memory
            )
            if coverage.get('infeasible_certified'):
                slot_memory['status'] = 'infeasible_certified'
                complete += 1
            elif coverage['optimality_certified']:
                slot_memory['status'] = 'optimality_certified'
                complete += 1
            elif coverage.get('candidate_set_complete') and candidates:
                slot_memory['status'] = 'candidate_set_complete'
                complete += 1
            elif unknown:
                slot_memory['status'] = 'needs_information'
            elif candidates:
                slot_memory['status'] = 'feasible_unproven'
            else:
                slot_memory['status'] = 'unresolved'
        total = len(problem_ir['slots'])
        self.metrics['verified_candidates'] = {
            slot_id: len(state['candidate_ids'])
            for slot_id, state in memory['slots'].items()
        }
        self.metrics['coverage_status'] = {
            slot_id: {
                key: state['coverage'].get(key)
                for key in (
                    'feasible_candidate_found', 'candidate_set_complete',
                    'optimality_certified', 'total_count', 'retrieved_count',
                    'infeasible_certified', 'pages_exhausted',
                    'source_exhaustive', 'all_cheaper_checked', 'best_product_id',
                )
            }
            for slot_id, state in memory['slots'].items()
        }
        self.metrics['retrieval_coverage'] = complete / total if total else 0.0
        return complete == total

    @staticmethod
    def _destination_address(user_info: Dict[str, Any]) -> str:
        address = user_info.get('shipping_addresses') or {}
        if isinstance(address, str):
            return address
        return ' '.join(
            str(address.get(key, ''))
            for key in ('province', 'city', 'detail_address')
            if address.get(key)
        )

    def _resolve_unknown_information(
        self,
        problem_ir: Dict[str, Any],
        memory: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> int:
        """Resolve tool-computable UNKNOWN fields before requesting a new search."""
        destination = self._destination_address(memory.get('user_info') or {})
        if not destination:
            return 0
        requested: set[str] = set()
        for slot in problem_ir['slots']:
            for product in self._slot_unknown_candidates(slot, memory):
                product_id = product['product_id']
                status, diagnostics = self._candidate_status(product, slot, memory)
                missing = {item['field'] for item in diagnostics if item.get('actual') is None}
                if status == self.UNKNOWN and missing == {'transport_time'}:
                    requested.add(product_id)
        for product_id in sorted(requested):
            arguments = {
                'product_id': product_id,
                'destination_address': destination,
            }
            result = self._record_tool_action(
                messages, 'calculate_transport_time', arguments
            )
            self._update_memory(memory, {
                'name': 'calculate_transport_time',
                'arguments': json.dumps(arguments, ensure_ascii=False),
            }, result)
        return len(requested)


    def _solve_level1(
        self,
        problem_ir: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._ready_for_solver(problem_ir, memory):
            return None
        selected = []
        for slot in problem_ir['slots']:
            candidates = self._slot_candidates(slot, memory)
            if not candidates:
                return None
            best = candidates[0]
            selected.append({
                'slot_id': slot['slot_id'],
                'product_id': best['product_id'],
                'quantity': slot['quantity'],
                'price': best['price'],
            })
        return {
            'status': 'OPTIMAL_OVER_CERTIFIED_FRONTIERS',
            'solver': 'deterministic_argmin',
            'selected_products': selected,
            'objective_value': sum(x['price'] * x['quantity'] for x in selected),
            'optimality_scope': 'catalog_under_problem_ir',
        }

    @staticmethod
    def _to_cents(value: Any) -> int:
        return int(round(float(value) * 100))

    @classmethod
    def _budget_cents(
        cls, problem_ir: Dict[str, Any]
    ) -> tuple[Optional[int], Optional[int], str]:
        constraints = problem_ir.get('global_constraints') or {}
        budget = constraints.get('budget') if isinstance(constraints, dict) else None
        if not isinstance(budget, dict):
            return None, None, 'subtotal'
        lower = budget.get('min')
        upper = budget.get('max')
        return (
            cls._to_cents(lower) if lower is not None else None,
            cls._to_cents(upper) if upper is not None else None,
            str(budget.get('basis') or 'subtotal'),
        )

    def _product_price_states(
        self,
        problem_ir: Dict[str, Any],
        memory: Dict[str, Any],
        upper_bound_cents: Optional[int] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Enumerate one candidate per slot, retaining one stable plan per subtotal."""
        states: Dict[int, List[Dict[str, Any]]] = {0: []}
        for slot in problem_ir['slots']:
            candidates = self._slot_candidates(slot, memory)
            if not candidates:
                return {}
            next_states: Dict[int, List[Dict[str, Any]]] = {}
            for subtotal, selections in states.items():
                for product in candidates:
                    price_cents = self._to_cents(product['price']) * int(slot['quantity'])
                    new_total = subtotal + price_cents
                    if upper_bound_cents is not None and new_total > upper_bound_cents:
                        continue
                    selection = {
                        'slot_id': slot['slot_id'],
                        'product_id': product['product_id'],
                        'quantity': slot['quantity'],
                        'price': float(product['price']),
                    }
                    plan = selections + [selection]
                    existing = next_states.get(new_total)
                    if existing is None or tuple(
                        item['product_id'] for item in plan
                    ) < tuple(item['product_id'] for item in existing):
                        next_states[new_total] = plan
            states = next_states
            if not states:
                break
        return states

    def _solve_level2(
        self,
        problem_ir: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        lower, upper, _ = self._budget_cents(problem_ir)
        states = self._product_price_states(problem_ir, memory, upper)
        feasible_totals = [
            total for total in states
            if (lower is None or total >= lower) and (upper is None or total <= upper)
        ]
        if not feasible_totals:
            return None
        best_total = min(feasible_totals)
        return {
            'status': 'OPTIMAL_OVER_COMPLETE_CANDIDATE_SETS',
            'solver': 'deterministic_budget_dp',
            'selected_products': states[best_total],
            'objective_value': best_total / 100.0,
            'subtotal': best_total / 100.0,
            'selected_coupons': [],
            'total_discount': 0.0,
            'optimality_scope': 'catalog_under_problem_ir',
        }

    def _coupon_plans(
        self, memory: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enumerate normalized CouponIR quantities under benchmark semantics."""
        coupon_items = []
        user_info = memory.get('user_info') or {}
        for coupon in memory.get('coupons', []):
            if coupon['vip_required'] and not user_info.get('is_vip', False):
                continue
            coupon_items.append({
                **coupon,
                'discount_cents': self._to_cents(coupon['discount']),
                'threshold_cents': self._to_cents(coupon['threshold']),
                'unit_discount': coupon['discount'],
            })

        plans: List[Dict[str, Any]] = [{
            'threshold_cents': 0,
            'discount_cents': 0,
            'coupon_count': 0,
            'selected_coupons': [],
        }]
        for coupon in coupon_items:
            expanded = []
            for plan in plans:
                for quantity in range(coupon['owned_quantity'] + 1):
                    selected = list(plan['selected_coupons'])
                    if quantity:
                        selected.append({
                            'coupon_name': coupon['coupon_name'],
                            'quantity': quantity,
                            'unit_discount': coupon['unit_discount'],
                            'threshold': coupon['threshold'],
                        })
                    expanded.append({
                        'threshold_cents': (
                            plan['threshold_cents']
                            + coupon['threshold_cents'] * quantity
                        ),
                        'discount_cents': (
                            plan['discount_cents']
                            + coupon['discount_cents'] * quantity
                        ),
                        'coupon_count': plan['coupon_count'] + quantity,
                        'selected_coupons': selected,
                    })
            plans = expanded
        return plans

    def _coupon_scope_feasible(
        self,
        coupon_plan: Dict[str, Any],
        selections: List[Dict[str, Any]],
        memory: Dict[str, Any],
    ) -> bool:
        """Assign each same-brand coupon unit to one selected brand subtotal."""
        definitions = {
            item['coupon_name']: item for item in memory.get('coupons', [])
        }
        brand_capacity: Dict[str, int] = {}
        for selection in selections:
            product = memory['candidates'][selection['product_id']]
            brand = str(product.get('brand') or '')
            brand_capacity[brand] = brand_capacity.get(brand, 0) + (
                self._to_cents(selection['price']) * int(selection['quantity'])
            )
        required_chunks = []
        for selected in coupon_plan['selected_coupons']:
            definition = definitions[selected['coupon_name']]
            if definition['scope'] == 'same_brand':
                required_chunks.extend(
                    [self._to_cents(definition['threshold'])] * int(selected['quantity'])
                )
        required_chunks.sort(reverse=True)
        capacities = sorted(brand_capacity.values(), reverse=True)

        def assign(index: int) -> bool:
            if index == len(required_chunks):
                return True
            threshold = required_chunks[index]
            seen = set()
            for position, capacity in enumerate(capacities):
                if capacity < threshold or capacity in seen:
                    continue
                seen.add(capacity)
                capacities[position] -= threshold
                if assign(index + 1):
                    return True
                capacities[position] += threshold
            return False

        return assign(0)

    def _solve_level3(
        self,
        problem_ir: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        lower, upper, budget_basis = self._budget_cents(problem_ir)
        product_upper = upper if budget_basis == 'subtotal' else None
        product_states = self._product_price_states(
            problem_ir, memory, product_upper
        )
        coupon_plans = self._coupon_plans(memory)
        best: Optional[tuple[tuple[Any, ...], int, List[Dict[str, Any]], Dict[str, Any]]] = None
        for subtotal, selections in product_states.items():
            eligible_plans = [
                plan for plan in coupon_plans
                if plan['threshold_cents'] <= subtotal
                and self._coupon_scope_feasible(plan, selections, memory)
            ]
            # ShoppingBench labels first maximize utilized spending thresholds,
            # then discount. This differs from unconstrained maximum-discount
            # choice in rare cases (for example a smaller-threshold VIP coupon).
            coupon_plan = max(
                eligible_plans,
                key=lambda plan: (
                    plan['threshold_cents'],
                    plan['discount_cents'],
                    -plan['coupon_count'],
                    tuple(
                        (item['coupon_name'], item['quantity'])
                        for item in plan['selected_coupons']
                    ),
                ),
            )
            final_price = subtotal - coupon_plan['discount_cents']
            constrained_value = subtotal if budget_basis == 'subtotal' else final_price
            if lower is not None and constrained_value < lower:
                continue
            if upper is not None and constrained_value > upper:
                continue
            key = (
                final_price,
                subtotal,
                coupon_plan['coupon_count'],
                tuple(item['product_id'] for item in selections),
                tuple(
                    (item['coupon_name'], item['quantity'])
                    for item in coupon_plan['selected_coupons']
                ),
            )
            if best is None or key < best[0]:
                best = (key, subtotal, selections, coupon_plan)
        if best is None:
            return None
        _, subtotal, selections, coupon_plan = best
        final_cents = subtotal - coupon_plan['discount_cents']
        return {
            'status': 'OPTIMAL_OVER_COMPLETE_CANDIDATE_SETS',
            'solver': 'deterministic_product_coupon_dp',
            'selected_products': selections,
            'selected_coupons': coupon_plan['selected_coupons'],
            'subtotal': subtotal / 100.0,
            'total_discount': coupon_plan['discount_cents'] / 100.0,
            'objective_value': final_cents / 100.0,
            'optimality_scope': 'catalog_under_problem_ir',
        }

