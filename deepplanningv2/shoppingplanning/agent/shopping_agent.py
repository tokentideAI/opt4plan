"""
Custom Agent implementation - Framework-independent

Uses universal LLM calling for multiple providers
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Lock

try:
    from .call_llm import call_llm
    from .planning_core import StructuredPlanningMixin
    from .prompts import prompt_lib
except ImportError:
    from call_llm import call_llm
    from planning_core import StructuredPlanningMixin
    from prompts import prompt_lib




class ShoppingFnAgent(StructuredPlanningMixin):
    """Structured shopping planner with an LLM parser and deterministic execution."""

    SOLVER_NAMES = {
        1: 'deterministic_argmin',
        2: 'deterministic_budget_dp',
        3: 'deterministic_product_coupon_dp',
    }

    READ_ONLY_TOOL_NAMES = {
        'search_products',
        'filter_by_brand',
        'filter_by_color',
        'filter_by_size',
        'filter_by_applicable_coupons',
        'filter_by_range',
        'sort_products',
        'get_product_details',
        'get_user_info',
        'calculate_transport_time',
    }

    ALLOWED_FIELDS = {
        'name', 'price', 'brand', 'color', 'size', 'stock_quantity',
        'suitable_season', 'target_demographic', 'sales_volume.monthly',
        'sales_volume.total', 'rating.average_score', 'rating.total_reviews',
        'rating.distribution.1_star', 'rating.distribution.2_star',
        'rating.distribution.3_star', 'rating.distribution.4_star',
        'rating.distribution.5_star', 'transport_time',
    }
    ALLOWED_OPS = {'==', '!=', '<', '<=', '>', '>=', 'contains'}
    SAT = 'SAT'
    UNSAT = 'UNSAT'
    UNKNOWN = 'UNKNOWN'
    COVERAGE_READY = 'READY'
    COVERAGE_INFEASIBLE = 'INFEASIBLE_CERTIFIED'
    COVERAGE_INCOMPLETE = 'INCOMPLETE'


    def __init__(self,
                 model: str | None = None,
                 tool_schema_path: str | None = None,
                 sample_id: str | None = None,
                 database_base_path: str | None = None,
                 level: int = 1) -> None:
        """
        Initialize Agent
        
        Args:
            model: Model name (must exist in models_config.json)
            tool_schema_path: Path to tool schema JSON file
            sample_id: Sample ID for database path resolution
            database_base_path: Base path to database directory
        """
        self._load_env_from_dotenv()

        self.model = model or os.getenv("TOOLS_AGENT_MODEL", "qwen-plus")
        if level not in (1, 2, 3):
            raise ValueError(f'unsupported shopping level: {level}')
        self.level = level
        self.metrics: Dict[str, Any] = {
            'parser_success': False,
            'parser_attempts': 0,
            'retrieval_calls': 0,
            'retrieved_products': 0,
            'product_details_requested': 0,
            'retrieval_budget': {},
            'verified_candidates': {},
            'retrieval_coverage': 0.0,
            'coverage_status': {},
            'solver_used': None,
            'solver_status': None,
            'solver_runtime': 0.0,
            'realization_actions': 0,
            'verification_success': False,
            'fallback_used': False,
        }
        domain_root = Path(__file__).resolve().parent.parent
        default_schema = domain_root / 'tools' / 'shopping_tool_schema.json'
        self.tool_schema_path = tool_schema_path or os.getenv("SHOPPING_SCHEMA_PATH", str(default_schema))

        self.sample_id = sample_id
        if database_base_path:
            self.database_base_path = Path(database_base_path)
        else:
            # Default path: ShoppingBench/database
            self.database_base_path = domain_root / 'database'

        self.tool_config = self._build_tool_config()
        self.tools_schema = self._load_tool_schemas()
        self.openai_tools = self._build_openai_tools(self.tools_schema)
        self.tool_instances = self._load_tool_instances()

        if not Path(self.tool_schema_path).exists():
            raise FileNotFoundError(f"Tool schema not found: {self.tool_schema_path}")

    def _build_tool_config(self) -> Dict[str, Any]:
        """
        Build tool configuration with database path.
        All shopping tools use the same products.jsonl file, simplifying the logic.
        """
        cfg = {}
        if self.sample_id is not None:
            # Shopping scenario database path structure: database/case_{sample_id}/products.jsonl
            db_path = self.database_base_path / f'case_{self.sample_id}'
            
            if db_path.exists():
                cfg['database_path'] = str(db_path)
            else:
                if os.getenv('DEBUG_TOOLS') == '1':
                    print(f"[ShoppingFnAgent] WARN: Database not found for case {self.sample_id}: {db_path}")
        return cfg
    
    def _load_tool_instances(self) -> Dict[str, Any]:
        """
        Dynamically load tool instances from TOOL_REGISTRY.
        
        Tool registration mechanism:
        1. Tool classes use the @register_tool('tool_name') decorator
        2. The decorator executes at class definition time, registering the tool class to base_shopping_tool.TOOL_REGISTRY
        3. When importing the tools package, __init__.py imports all tool modules, triggering decorator execution
        4. Retrieve registered tool classes from TOOL_REGISTRY and instantiate them
        """
        instances: Dict[str, Any] = {}

        tools_dir = Path(__file__).resolve().parent.parent / 'tools'
        # Add tools_dir to sys.path to enable 'from base_shopping_tool import ...' in tool files
        sys.path.insert(0, str(tools_dir))
        sys.path.insert(0, str(tools_dir.parent))

        # Import tools package to trigger @register_tool decorator execution for all tool modules
        # tools/__init__.py imports all tool modules, and decorators register tool classes to TOOL_REGISTRY
        try:
            import tools  # noqa: F401
        except Exception as e:
            if os.getenv('DEBUG_TOOLS') == '1':
                print(f"[ShoppingFnAgent] WARN: import tools failed: {e}")
            return instances

        # Get TOOL_REGISTRY from base_shopping_tool module
        try:
            import base_shopping_tool  # type: ignore
            tool_registry = getattr(base_shopping_tool, 'TOOL_REGISTRY', None)
            if tool_registry is None:
                if os.getenv('DEBUG_TOOLS') == '1':
                    print("[ShoppingFnAgent] WARN: TOOL_REGISTRY not found in base_shopping_tool")
                return instances
        except Exception as e:
            if os.getenv('DEBUG_TOOLS') == '1':
                print(f"[ShoppingFnAgent] WARN: import base_shopping_tool failed: {e}")
            return instances

        if not tool_registry:
            print("[ShoppingFnAgent] WARN: TOOL_REGISTRY is empty. No tools were registered.")
            return instances

        # Create tool instances from TOOL_REGISTRY
        tool_cfg = self.tool_config
        for tool_name, tool_cls in tool_registry.items():
            try:
                inst = tool_cls(cfg=tool_cfg)
                instances[tool_name] = inst
            except Exception as e:
                if os.getenv('DEBUG_TOOLS') == '1':
                    print(f"[ShoppingFnAgent] WARN: Failed to instantiate tool '{tool_name}': {e}")
                continue

        return instances

    def _load_env_from_dotenv(self) -> None:
        """
        Load environment variables from .env file
        
        Searches for .env in the following order:
        1. Domain directory (shoppingplanning/)
        2. Project root (parent of domain)
        """
        try:
            # Try domain directory first
            domain_root = Path(__file__).resolve().parent.parent
            domain_dotenv = domain_root / '.env'
            
            # Try project root
            project_root = domain_root.parent
            project_dotenv = project_root / '.env'
            
            # Use project root .env if it exists, otherwise domain .env
            dotenv_path = project_dotenv if project_dotenv.exists() else domain_dotenv
            
            if not dotenv_path.exists():
                return
            
            for line in dotenv_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and (key not in os.environ):
                    os.environ[key] = val
        except Exception:
            pass

    def _load_tool_schemas(self) -> List[Dict[str, Any]]:
        """Load tool schemas from JSON file"""
        with open(self.tool_schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_openai_tools(self, schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build OpenAI tools format
        - If schema is already {type:function, function:{...}}, use as-is
        - Otherwise wrap as function definition
        """
        tools: List[Dict[str, Any]] = []
        for s in schemas:
            if isinstance(s, dict) and s.get('type') == 'function' and isinstance(s.get('function'), dict):
                tools.append(s)
        return tools

    def _exec_tool(self, name: str, arguments_json: str) -> str:
        """Execute tool call"""
        if name in self.READ_ONLY_TOOL_NAMES:
            self.metrics['retrieval_calls'] += 1
        elif name in {
            'add_product_to_cart', 'delete_product_from_cart',
            'add_coupon_to_cart', 'delete_coupon_from_cart',
        }:
            self.metrics['realization_actions'] += 1
        inst = self.tool_instances.get(name)
        if not inst:
            return json.dumps({"error": f"tool '{name}' not found"}, ensure_ascii=False)
        try:
            res = inst.call(arguments_json)  # Pass raw JSON string
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
        """Call LLM with unified handling for all models"""
        return call_llm(
            config_name=self.model,
            messages=messages,
            tools=tools
        )

    def _detect_tool_calls(self, assistant_message) -> List[Dict[str, Any]]:
        """Detect and normalize tool calls"""
        tool_calls = getattr(assistant_message, 'tool_calls', None)
        calls: List[Dict[str, Any]] = []
        if not tool_calls:
            return calls
        
        for idx, tc in enumerate(tool_calls):
            try:
                # Generate unique ID if not provided by the model
                tool_call_id = tc.id
                if tool_call_id is None or not tool_call_id:
                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                
                calls.append({
                    'id': tool_call_id,
                    'name': tc.function.name,
                    'arguments': tc.function.arguments,
                })
            except Exception:
                continue
        
        return calls

    def _message_to_dict(self, message, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize an SDK assistant message for history persistence."""
        result = {'role': 'assistant', 'content': message.content or ''}
        if getattr(message, 'reasoning_content', None):
            result['reasoning_content'] = message.reasoning_content
        if calls:
            result['tool_calls'] = [
                {
                    'id': call['id'],
                    'type': 'function',
                    'function': {
                        'name': call['name'],
                        'arguments': call['arguments'],
                    },
                }
                for call in calls
            ]
        return result

    def _record_tool_action(
        self,
        messages: List[Dict[str, Any]],
        name: str,
        arguments: Dict[str, Any],
        compact_result: bool = False,
    ) -> str:
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        arguments_json = json.dumps(arguments, ensure_ascii=False)
        messages.append({
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {'name': name, 'arguments': arguments_json},
            }],
        })
        result = self._exec_tool(name, arguments_json)
        message_result = (
            self._compact_tool_result(name, result) if compact_result else result
        )
        messages.append({
            'role': 'tool', 'tool_call_id': call_id, 'content': message_result,
        })
        return result

    @staticmethod
    def _compact_tool_result(name: str, result: str) -> str:
        """Keep deterministic retrieval trajectories lightweight."""
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return result
        if not isinstance(payload, dict) or payload.get('error'):
            return result
        if name == 'get_product_details':
            products = payload.get('products') or []
            return json.dumps({
                'product_count': len(products),
                'product_ids': [item.get('product_id') for item in products],
                'details_stored_in_planner_memory': True,
            }, ensure_ascii=False)
        id_key = next((
            key for key in (
                'filtered_products_ids', 'sorted_product_ids', 'product_ids'
            ) if isinstance(payload.get(key), list)
        ), None)
        if id_key:
            ids = payload[id_key]
            compact = {
                key: payload.get(key) for key in (
                    'input_count', 'total_count', 'returned_count', 'is_exhaustive'
                ) if key in payload
            }
            compact.update({
                'id_count': len(ids),
                'sample_product_ids': ids[:10],
                'full_ids_consumed_by_deterministic_retriever': True,
            })
            return json.dumps(compact, ensure_ascii=False)
        return result

    def _get_cart(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Observe cart state through the benchmark's official tool interface."""
        result = self._record_tool_action(messages, 'get_cart_info', {})
        payload = json.loads(result)
        if not isinstance(payload, dict) or payload.get('error'):
            raise RuntimeError(f'get_cart_info failed: {payload}')
        return payload

    def _realize_solution(
        self,
        solution: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compile a solver result into deterministic cart mutations."""
        targets: Dict[str, int] = {}
        for selection in solution['selected_products']:
            product_id = selection['product_id']
            targets[product_id] = targets.get(product_id, 0) + selection['quantity']

        cart = self._get_cart(messages)
        for coupon in cart.get('used_coupons', []):
            self._record_tool_action(messages, 'delete_coupon_from_cart', {
                'coupon_name': coupon['coupon_name'],
                'quantity': coupon.get('quantity', 1),
            })
        for item in cart.get('items', []):
            current = int(item.get('quantity', 0))
            target = targets.get(item.get('product_id'), 0)
            if current > target:
                self._record_tool_action(messages, 'delete_product_from_cart', {
                    'product_id': item['product_id'],
                    'quantity': current - target,
                })

        current_quantities = {
            item['product_id']: int(item.get('quantity', 0))
            for item in self._get_cart(messages).get('items', [])
        }
        for product_id, target in targets.items():
            missing = target - current_quantities.get(product_id, 0)
            if missing > 0:
                self._record_tool_action(messages, 'add_product_to_cart', {
                    'product_id': product_id,
                    'quantity': missing,
                })
        for coupon in solution.get('selected_coupons', []):
            self._record_tool_action(messages, 'add_coupon_to_cart', {
                'coupon_name': coupon['coupon_name'],
                'quantity': coupon['quantity'],
            })
        return self._get_cart(messages)

    def _verify_solution(
        self,
        problem_ir: Dict[str, Any],
        solution: Dict[str, Any],
        final_cart: Dict[str, Any],
        memory: Dict[str, Any],
        level: int = 1,
    ) -> Dict[str, Any]:
        expected: Dict[str, int] = {}
        for selection in solution['selected_products']:
            product_id = selection['product_id']
            expected[product_id] = expected.get(product_id, 0) + selection['quantity']
        actual = {
            item['product_id']: int(item.get('quantity', 0))
            for item in final_cart.get('items', [])
        }
        errors = []
        if actual != expected:
            errors.append({'cart_mismatch': {'expected': expected, 'actual': actual}})
        slots = {str(slot['slot_id']): slot for slot in problem_ir['slots']}
        seen_slots = set()
        for selection in solution['selected_products']:
            product = memory['candidates'].get(selection['product_id'])
            slot_key = str(selection['slot_id'])
            if slot_key not in slots or slot_key in seen_slots:
                errors.append({'slot_mismatch': selection})
                continue
            seen_slots.add(slot_key)
            slot = slots[slot_key]
            if not product:
                errors.append({'missing_product_details': selection})
                continue
            status, diagnostics = self._candidate_status(product, slot, memory)
            if status != self.SAT:
                errors.append({
                    'constraint_mismatch': selection,
                    'status': status,
                    'diagnostics': diagnostics,
                })
                continue
            feasible = self._slot_candidates(slot, memory)
            feasible_ids = {product['product_id'] for product in feasible}
            if selection['product_id'] not in feasible_ids:
                errors.append({
                    'product_outside_feasible_candidates': {
                        'selected': selection,
                    }
                })
            elif level == 1 and feasible[0]['product_id'] != selection['product_id']:
                errors.append({
                    'not_optimal_over_retrieved_candidates': {
                        'selected': selection,
                        'cheapest_product_id': feasible[0]['product_id'],
                    }
                })
        if seen_slots != set(slots):
            errors.append({'missing_slots': sorted(set(slots) - seen_slots)})

        subtotal = sum(
            self._to_cents(selection['price']) * int(selection['quantity'])
            for selection in solution['selected_products']
        )
        expected_coupons = {
            item['coupon_name']: int(item['quantity'])
            for item in solution.get('selected_coupons', [])
        }
        actual_coupons = {
            item['coupon_name']: int(item.get('quantity', 0))
            for item in final_cart.get('used_coupons', [])
        }
        if actual_coupons != expected_coupons:
            errors.append({
                'coupon_mismatch': {
                    'expected': expected_coupons,
                    'actual': actual_coupons,
                }
            })

        discount = 0
        threshold = 0
        coupon_ir = {
            item['coupon_name']: item for item in memory.get('coupons', [])
        }
        for coupon in solution.get('selected_coupons', []):
            definition = coupon_ir.get(coupon['coupon_name'])
            quantity = int(coupon['quantity'])
            if definition is None or quantity > definition['owned_quantity']:
                errors.append({'invalid_coupon': coupon})
                continue
            if definition['vip_required'] \
                    and not (memory.get('user_info') or {}).get('is_vip', False):
                errors.append({'vip_coupon_for_non_vip': coupon})
            discount += self._to_cents(definition['discount']) * quantity
            threshold += self._to_cents(definition['threshold']) * quantity
        if threshold > subtotal:
            errors.append({
                'coupon_threshold_exceeded': {
                    'required': threshold / 100.0,
                    'subtotal': subtotal / 100.0,
                }
            })
        if not self._coupon_scope_feasible(
            {'selected_coupons': solution.get('selected_coupons', [])},
            solution['selected_products'],
            memory,
        ):
            errors.append({'same_brand_coupon_scope_infeasible': True})

        final_price = subtotal - discount
        if abs(float(solution.get('objective_value', 0)) - final_price / 100.0) > 1e-6:
            errors.append({
                'objective_mismatch': {
                    'solution': solution.get('objective_value'),
                    'calculated': final_price / 100.0,
                }
            })
        cart_total = (final_cart.get('summary') or {}).get('total_price')
        if cart_total is None or abs(float(cart_total) - final_price / 100.0) > 1e-6:
            errors.append({
                'cart_total_mismatch': {
                    'expected': final_price / 100.0,
                    'actual': cart_total,
                }
            })
        lower, upper, basis = self._budget_cents(problem_ir)
        budget_value = subtotal if basis == 'subtotal' else final_price
        if lower is not None and budget_value < lower:
            errors.append({'budget_below_minimum': budget_value / 100.0})
        if upper is not None and budget_value > upper:
            errors.append({'budget_above_maximum': budget_value / 100.0})
        return {
            'success': not errors,
            'errors': errors,
            'optimality_scope': (
                'catalog_under_problem_ir'
                if (
                    self._ready_for_solver(problem_ir, memory)
                    if level == 1 else all(
                        self._slot_enumeration_complete(slot, memory)
                        for slot in problem_ir['slots']
                    )
                )
                else 'retrieved_verified_candidates'
            ),
        }

    def _run_react_repair(
        self,
        messages: List[Dict[str, Any]],
        max_llm_calls: int,
        save_messages: bool,
        messages_file: Optional[Path],
        repair_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self.metrics['fallback_used'] = True
        context = json.dumps(repair_context or {}, ensure_ascii=False, default=str)
        messages.append({
            'role': 'user',
            'content': (
                'Inspect the current cart and complete the original request. '
                'You may now use all tools. Correct any missing, extra, or suboptimal items, '
                'then finish with a final answer and no tool calls. Focus first on the slots '
                f'identified by this structured repair context:\n{context}'
            ),
        })
        for step in range(1, max_llm_calls + 1):
            response = self._call_llm(messages=messages, tools=self.openai_tools)
            message = response.choices[0].message
            calls = self._detect_tool_calls(message)
            messages.append(self._message_to_dict(message, calls))
            if not calls:
                if save_messages and messages_file:
                    self._save_messages(
                        messages, messages_file, step, 'Fallback ReAct completed'
                    )
                return messages
            for call in calls:
                result = self._exec_tool(call['name'], call['arguments'])
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': result})
        if save_messages and messages_file:
            self._save_messages(
                messages, messages_file, max_llm_calls,
                'Fallback ReAct reached call limit',
            )
        return messages

    def run(self, user_query: str, system_prompt: str | None = None, max_llm_calls: int = 100,
            save_messages: bool = True, messages_output_dir: str | None = None,
            sample_id: str | None = None, level: Optional[int] = None) -> List[Any]:
        """Parse → retrieve → deterministic solve → realize → verify → targeted fallback."""
        active_level = self.level if level is None else level
        if active_level not in (1, 2, 3):
            raise ValueError(f'unsupported shopping level: {active_level}')

        messages_file: Optional[Path] = None
        if save_messages:
            if sample_id:
                directory = self.database_base_path / f'case_{sample_id}'
            else:
                directory = Path(messages_output_dir or (
                    Path(__file__).resolve().parent.parent / 'result' / 'messages'
                ))
            directory.mkdir(parents=True, exist_ok=True)
            messages_file = directory / ('messages.json' if sample_id else (
                f'messages_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            ))

        original_messages: List[Dict[str, Any]] = (
            ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
            + [{'role': 'user', 'content': user_query}]
        )
        if save_messages and messages_file:
            self._save_messages(original_messages, messages_file, 0, 'Initial messages')

        messages = list(original_messages)
        user_result = self._record_tool_action(messages, 'get_user_info', {})
        try:
            user_info = json.loads(user_result)
            if not isinstance(user_info, dict) or user_info.get('error'):
                user_info = {}
        except json.JSONDecodeError:
            user_info = {}

        try:
            self.problem_ir = self._parse_query(user_query, user_info, active_level)
            self.planning_memory = self._init_memory(self.problem_ir, user_info)
        except Exception as error:
            return self._run_react_repair(
                messages, max_llm_calls, save_messages, messages_file,
                repair_context={
                    'reason': 'parser_failed',
                    'error': str(error),
                },
            )

        coverage = self._retrieve_and_certify(
            active_level, self.problem_ir, self.planning_memory, messages
        )
        self.coverage = coverage
        if save_messages and messages_file:
            self._save_messages(
                messages, messages_file, 1, 'Deterministic retrieval and coverage'
            )
        if coverage['status'] == self.COVERAGE_INCOMPLETE:
            return self._run_react_repair(
                messages, max_llm_calls, save_messages, messages_file,
                repair_context={
                    'reason': 'retrieval_incomplete',
                    'problem_ir': self.problem_ir,
                    'slot_states': self.planning_memory['slots'],
                },
            )
        if coverage['status'] == self.COVERAGE_INFEASIBLE:
            self.metrics['solver_status'] = 'infeasible_certified'
            messages.append({
                'role': 'assistant',
                'content': json.dumps({
                    'status': 'failure',
                    'items': [],
                    'reason': 'no_feasible_product',
                    'coverage': coverage['status'],
                }, ensure_ascii=False),
            })
            if save_messages and messages_file:
                self._save_messages(
                    messages, messages_file, 2, 'Certified infeasible result'
                )
            return messages

        solver_started = time.time()
        if active_level == 1:
            self.solution = self._solve_level1(
                self.problem_ir, self.planning_memory
            )
        elif active_level == 2:
            self.solution = self._solve_level2(
                self.problem_ir, self.planning_memory
            )
        else:
            self.solution = self._solve_level3(
                self.problem_ir, self.planning_memory
            )
        self.metrics['solver_runtime'] = time.time() - solver_started
        self.metrics['solver_used'] = (
            self.solution.get('solver') if self.solution
            else self.SOLVER_NAMES[active_level]
        )
        self.metrics['solver_status'] = (
            self.solution.get('status') if self.solution else 'infeasible'
        )
        if not self.solution:
            messages.append({
                'role': 'assistant',
                'content': json.dumps({
                    'status': 'failure',
                    'items': [],
                    'reason': 'constraints_infeasible',
                }, ensure_ascii=False),
            })
            if save_messages and messages_file:
                self._save_messages(
                    messages, messages_file, 2, 'Deterministic solver infeasible'
                )
            return messages

        final_cart = self._realize_solution(self.solution, messages)
        self.verification = self._verify_solution(
            self.problem_ir, self.solution, final_cart, self.planning_memory,
            level=active_level,
        )
        self.metrics['verification_success'] = self.verification['success']
        if not self.verification['success']:
            return self._run_react_repair(
                messages, max_llm_calls, save_messages, messages_file,
                repair_context={
                    'reason': 'verification_failed',
                    'problem_ir': self.problem_ir,
                    'selected_products': self.solution['selected_products'],
                    'current_cart': final_cart,
                    'verification_errors': self.verification['errors'],
                },
            )

        items = []
        for selection in self.solution['selected_products']:
            product = self.planning_memory['candidates'][selection['product_id']]
            items.append({
                'slot_id': selection['slot_id'],
                'product_id': selection['product_id'],
                'product_name': product.get('name', ''),
                'quantity': selection['quantity'],
                'unit_price': selection['price'],
            })
        final_response = {
            'status': 'success',
            'items': items,
            'total_price': self.solution['objective_value'],
        }
        if active_level == 3:
            final_response.update({
                'coupons': self.solution.get('selected_coupons', []),
                'subtotal': self.solution.get('subtotal'),
                'total_discount': self.solution.get('total_discount'),
            })
        messages.append({
            'role': 'assistant',
            'content': json.dumps(final_response, ensure_ascii=False),
        })
        if save_messages and messages_file:
            self._save_messages(
                messages, messages_file, max_llm_calls,
                f'Verified deterministic Level-{active_level} solution'
            )
        return messages

    def _save_messages(self, messages: List[Any], filepath: Path, step: int, description: str):
        """Save messages to file"""
        serializable_messages = [m.model_dump() if hasattr(m, 'model_dump') else m for m in messages]
        save_data = {"step": step, "description": description, "messages": serializable_messages}
        planning_state = {
            key: getattr(self, key)
            for key in (
                'problem_ir', 'coverage', 'solution', 'verification', 'metrics',
            )
            if hasattr(self, key)
        }
        if hasattr(self, 'planning_memory'):
            memory = self.planning_memory
            planning_state['planning_memory'] = (
                memory if 'trace' in memory else {
                    'slots': memory.get('slots', {}),
                    'candidate_count': len(memory.get('candidates', {})),
                    'transport_times': memory.get('transport_times', {}),
                    'coupons': memory.get('coupons', []),
                }
            )
        if planning_state:
            save_data['planning_state'] = planning_state
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            thread_info = threading.current_thread().name
            print(f"  💾 [{thread_info}] Step {step}: {description} - Saved {len(messages)} messages")
        except Exception as e:
            thread_info = threading.current_thread().name
            print(f"  ⚠️  [{thread_info}] Failed to save messages: {e}")


def run_agent_inference(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    level: int = 1,
    workers: int = 10,
    max_llm_calls: int = 100,
    rerun_ids: Optional[List[int]] = None,
    agent_class=None,
) -> Dict[str, Any]:
    """
    Run agent inference (batch processing)
    
    Args:
        model: Configuration name from models_config.json
        test_data_path: Path to test data JSON file
        database_dir: Base path to database directory
        tool_schema_path: Path to tool schema JSON file
        system_prompt: System prompt for the agent
        workers: Number of parallel workers
        max_llm_calls: Maximum LLM calls per sample
        rerun_ids: Optional list of specific IDs to rerun. If None, run all samples.
    
    Returns:
        Results summary dict
    """
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Filter samples if rerun_ids is specified
    if rerun_ids is not None:
        rerun_ids_set = set(str(id) for id in rerun_ids)  # Convert to strings for comparison
        original_count = len(test_data)
        test_data = [s for s in test_data if str(s.get('id')) in rerun_ids_set]
        print(f"  🔄 Filtered {original_count} samples to {len(test_data)} samples for rerun")
        
        if len(test_data) == 0:
            print(f"  ⚠️  Warning: No samples found matching the specified IDs")
            return {
                'total': 0,
                'success': 0,
                'failed': 0,
                'elapsed_time': 0,
                'results': []
            }
    
    print(f"\n{'='*80}")
    print(f"Agent Inference")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Samples: {len(test_data)}")
    print(f"Workers: {workers}")
    print(f"{'='*80}\n")
    
    print_lock = Lock()
    results = []
    agent_type = agent_class or ShoppingFnAgent
    
    def process_sample(sample):
        sample_id = sample.get('id', 'unknown')
        query = sample.get('query', '')
        
        try:
            
            agent = agent_type(
                model=model,
                sample_id=str(sample_id),
                database_base_path=str(database_dir),
                tool_schema_path=str(tool_schema_path),
                level=level,
            )
            
            start_time = time.time()
            
            messages = agent.run(
                user_query=query,
                system_prompt=system_prompt,
                save_messages=True,
                sample_id=str(sample_id),
                max_llm_calls=max_llm_calls,
                level=level,
            )
            
            elapsed = time.time() - start_time
            
            result = {
                'id': sample_id,
                'query': query,
                'model': model,
                'messages': messages,
                'elapsed_time': elapsed,
                'execution_success': True,
                'metrics': agent.metrics,
            }
            
            with print_lock:
                print(f"✅ Sample {sample_id} completed in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            with print_lock:
                print(f"❌ Sample {sample_id} failed: {e}")
                import traceback
                traceback.print_exc()
            
            return {
                'id': sample_id,
                'query': query,
                'execution_success': False,
                'error': str(e),
            }
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in test_data]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    success_count = sum(1 for r in results if r['execution_success'])
    
    return {
        'total': len(results),
        'success': success_count,
        'failed': len(results) - success_count,
        'results': results
    }


if __name__ == '__main__':
    """Simple test"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen-plus', help='Configuration name from models_config.json')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3], help='Shopping level: 1, 2, or 3')
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent
    
    # Get system prompt for the specified level
    try:
        from .prompts import prompt_lib
    except ImportError:
        from prompts import prompt_lib
    
    system_prompt = getattr(prompt_lib, f'SYSTEM_PROMPT_level{args.level}', None)
    if system_prompt is None:
        raise ValueError(f"System prompt for level {args.level} not found")
    
    result = run_agent_inference(
        model=args.model,
        test_data_path=base_dir / 'data' / f'level_{args.level}_query_meta.json',
        database_dir=base_dir / 'database',
        tool_schema_path=base_dir / 'tools' / 'shopping_tool_schema.json',
        system_prompt=system_prompt,
        level=args.level,
        workers=2,
        max_llm_calls=100,
    )
    print(f"\nTest completed: {result['success']}/{result['total']} succeeded")
