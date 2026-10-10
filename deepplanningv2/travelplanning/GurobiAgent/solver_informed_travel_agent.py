"""Solver-informed travel-planning agent in one Python file.

Design
======
This module is intentionally self-contained at the agent layer. It keeps the
benchmark's existing `tools/` package, tool-schema JSON files, case databases,
and `models_config.json`, but consolidates all agent-side logic here:

    NL request
      -> ProblemIR parsing
      -> independent semantic audit
      -> structured evidence/state
      -> candidate construction
      -> solver-guided information acquisition
      -> global multi-day candidate assignment
      -> lazy route acquisition + exact temporal scheduling
      -> conflict diagnosis / targeted expansion
      -> deterministic verification
      -> deterministic rendering
      -> optional conflict-guided reflection
      -> solver-based feasibility restoration (minimum-violation fallback)

The controller, rather than the LLM, decides *what information is needed next*.
The LLM compiles/audits ProblemIR and may propose one constrained recovery action;
all accepted retrieval intents are deterministically compiled into AcquisitionRequest.
All official-tool arguments are built by deterministic protocol adapters.

Important protocol invariant
----------------------------
The solver never reads hidden case CSV files directly. Environment facts enter
planner state only through official tool calls. The only direct file access is
for loading the public tool schemas and model configuration.

This file is meant to live at:
    travelplanning/GurobiAgent/solver_informed_travel_agent.py

Typical run:
    python travelplanning/GurobiAgent/solver_informed_travel_agent.py \
        --model deepseek-flash --language zh --workers 2 \
        --max-llm-calls 30 --max-tool-calls 400 --ids 0 1 2 \
        --evaluate
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
import sys
import threading
import time
import uuid
import hashlib
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import openai

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    gp = None
    GRB = None
    HAS_GUROBI = False


# =============================================================================
# 1. Prompts
# =============================================================================

PROBLEM_IR_PROMPT = r"""You are a compiler from a travel-planning request into strict JSON.
Return JSON only. Do not solve the trip and do not invent constraints.

Schema:
{
  "trip": {
    "origin": "string or null",
    "destination": "string or null",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "travelers": 1,
    "rooms": 1,
    "budget_max": null,
    "currency": "RMB"
  },
  "transport": {
    "outbound": {
      "allowed_modes": ["flight", "train"],
      "constraints": [],
      "soft_preferences": [],
      "objective": null
    },
    "return": {
      "allowed_modes": ["flight", "train"],
      "constraints": [],
      "soft_preferences": [],
      "objective": null
    }
  },
  "hotel": {
    "constraints": [],
    "soft_preferences": [],
    "objective": null
  },
  "attractions": {
    "count": null,
    "selection": "default|top_k|all_matching|named",
    "required_names": [],
    "constraints": [],
    "soft_preferences": [],
    "objective": null
  },
  "restaurants": {
    "requirements": [
      {
        "required_name": null,
        "nearby_location": null,
        "constraints": [],
        "soft_preferences": [],
        "objective": null
      }
    ]
  },
  "global_constraints": [],
  "explicit_preferences": []
}

Each constraint must be:
{
  "field": "canonical semantic field name",
  "op": "==|!=|<|<=|>|>=|contains|in",
  "value": "JSON value",
  "source_text": "exact user phrase"
}

An objective, if explicitly requested, must be:
{
  "field": "price|duration|rating|distance|departure_time|arrival_time|...",
  "sense": "min|max",
  "source_text": "exact user phrase"
}

Rules:
1. Preserve strict inequalities exactly.
2. Extract only explicit requirements. Do not add preferences such as cheapest,
   earliest, highest-rated, or direct unless the user states them.
3. Do not infer hidden database facts.
4. A total trip budget belongs only in trip.budget_max.
5. If the query asks for top-k / exactly-k attractions, put k in attractions.count.
6. If the request specifies outbound and return modes separately, keep them separate.
7. Keep entity/location names exactly as written by the user.
8. travelers and rooms default to 1 only when the request omits them.
9. Use only these canonical fields in constraints:
   - transport: seat_class, train_type, manufacturer, direct, price, duration,
     departure_time, arrival_time;
   - hotel: star, services, brand, decoration_year, price, rating;
   - attraction: name, attraction_type, price, rating;
   - restaurant: name, tags, cuisine, price, rating, nearby_location.
10. Put every explicitly named must-visit attraction in required_names and use
    selection="named" unless another count/all-matching request is also present.
11. "all recommended <type> attractions" means selection="all_matching" plus
    an attraction_type equality constraint. "top three" means count=3,
    selection="top_k", objective rating/max.
12. Each distinct dining request must be a separate restaurants.requirements item.
    A restaurant near a place stores the exact place in nearby_location. A named
    restaurant stores it in required_name. Tags/services and cuisine stay in that
    requirement's constraints; never merge two dining requests together.
13. "newly renovated after 2025" means decoration_year > 2025, not >=.
14. If a train/flight must have enough seats, do not add a user constraint: the
    controller always enforces seats >= travelers.
15. Polite wording such as "最好", "希望", or "prefer" does not waive a concrete,
    objectively checkable requirement in this benchmark. Keep such predicates hard
    and encode superlatives as objectives. Use soft_preferences only when the user
    explicitly says a condition is optional/non-mandatory (for example "非必须",
    "不强求", "可有可无", "optional", or "nice to have").
16. Never emit placeholder constraint objects. If no constraint applies, use an
    empty list; every emitted constraint must have a non-empty field and op.
17. Numeric fields require numeric values. Qualitative phrases such as "价格合适",
    "质量高", or "comfortable" must not become numeric equality constraints.
16. "最高评分"/"最便宜"/"最早" denote an objective, never a constraint
    with value "max", "min", "highest", or "lowest".
17. Time constraint values are integer minutes since midnight (e.g. 06:00 = 360).
18. 24:00 is the legal inclusive endpoint 1440 for an end-of-day interval; do not
    rewrite it to 23:59.
"""

SEMANTIC_AUDIT_PROMPT = r"""You are a semantic auditor for a typed travel Constraint IR.
Compare the original request with the parsed IR. Do not plan the trip, use tools, infer
database facts, or add reasonable defaults. Check only whether the IR faithfully captures
the user's explicit meaning, especially:
- hard requirement vs soft preference vs optimization objective;
- earliest departure vs earliest arrival, shortest duration, cheapest, closest, highest;
- counts, all/top-k/named coverage, scope, and referenced locations/entities;
- distinct restaurant requirements that must not be merged.
- 24:00 is intentionally represented as 1440 and is not a semantic error.

Return JSON only:
{
  "verdict": "PASS" | "REPAIR",
  "issues": [
    {"kind": "missing|incorrect|modality|objective|reference|scope",
     "source_text": "exact phrase from the request", "reason": "short explanation"}
  ],
  "corrected_ir": null | {FULL PROBLEM IR}
}

If verdict is PASS, corrected_ir must be null. If verdict is REPAIR, return the full
corrected IR using the same schema. Every added or changed constraint/objective must carry
an exact source_text substring from the request. Never weaken words such as must/required/
必须/一定/务必 merely because a requirement may be infeasible.
"""

REFLECTION_PROMPT = r"""You are a conflict-guided reflection controller for a symbolic
travel planner. The deterministic solver and normal retrieval policy have stopped making
progress. Diagnose whether the remaining failure is a search-space coverage gap, search
exhaustion, semantic mismatch, or proven infeasibility. Do not create an itinerary, name
new entities, invent facts, relax explicit hard requirements, or call tools.

Return JSON only:
{
  "diagnosis": "COVERAGE_GAP" | "SEARCH_EXHAUSTED" | "SEMANTIC_MISMATCH" | "POSSIBLE_TRUE_INFEASIBILITY",
  "action": "KEEP" | "RECHECK_SEMANTICS" | "EXPAND_RECALL",
  "reason": "short conflict-grounded explanation",
  "corrected_ir": null | {FULL PROBLEM IR},
  "recall_expansions": [
    {
      "constraint_id": "one supplied recall-sensitive constraint id",
      "kind": "additional_anchor" | "attraction_type" | "unrestricted_attractions",
      "values": ["values selected only from supplied allowed values"]
    }
  ]
}

RECHECK_SEMANTICS is allowed only for a demonstrable parse/modality/objective/reference
error and must return a complete corrected IR grounded in the original request.
EXPAND_RECALL may only select supplied grounded anchors or official attraction types. KEEP
is mandatory when no novel legal action exists. You may flag possible true infeasibility,
but only deterministic code can issue or validate a formal certificate. Reflection proposes
retrieval intent; the deterministic controller compiles it.
"""

# =============================================================================
# 2. Small data structures
# =============================================================================

SAT = "SAT"
UNSAT = "UNSAT"
UNKNOWN = "UNKNOWN"

READY = "READY"
NEEDS_INFO = "NEEDS_INFO"
INFEASIBLE = "INFEASIBLE"
COVERAGE_GAP = "COVERAGE_GAP"
SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
EXACT_FEASIBLE = "EXACT_FEASIBLE"
RELAXED_FEASIBLE = "RELAXED_FEASIBLE"

QUERY_SUCCESS_NONEMPTY = "SUCCESS_NONEMPTY"
QUERY_SUCCESS_EMPTY = "SUCCESS_EMPTY"
QUERY_PARSE_ERROR = "PARSE_ERROR"
QUERY_ERROR = "ERROR"

# DeepPlanning official itinerary-structure rule used consistently by template
# construction and final verification.
RETURN_DAY_ATTRACTION_THRESHOLD = 14 * 60


@dataclass
class AcquisitionRequest:
    tool: str
    purpose: str
    args_hint: Dict[str, Any]
    key: str
    priority: int = 0


@dataclass
class EvidenceRecord:
    record_id: str
    kind: str
    source_tool: str
    raw: Dict[str, Any]
    canonical: Dict[str, Any]


@dataclass(frozen=True)
class QueryOutcome:
    status: str
    exhaustive: bool
    tool: str
    record_count: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class InfeasibilityCertificate:
    kind: str
    constraints: Tuple[str, ...]
    evidence_keys: Tuple[str, ...]
    complete_domains: Tuple[str, ...]
    lhs: Optional[float] = None
    rhs: Optional[float] = None
    explanation: str = ""


@dataclass
class ActivitySlot:
    slot_id: str
    day: int
    kind: str                 # attraction | lunch | dinner
    candidates: List[str]
    earliest: Optional[int] = None
    latest: Optional[int] = None
    requirement_index: Optional[int] = None


@dataclass
class DayTemplate:
    day: int
    start_anchor: str
    end_anchor: str
    start_time: int
    end_deadline: int
    slots: List[ActivitySlot]


@dataclass
class SolveResult:
    status: str
    reason: str = ""
    solution: Optional[Dict[str, Any]] = None
    missing: List[AcquisitionRequest] = field(default_factory=list)
    conflict: Dict[str, Any] = field(default_factory=dict)
    certificate: Optional[InfeasibilityCertificate] = None


@dataclass(frozen=True)
class ConstraintIR:
    """Typed, executable constraint shared by propagation and verification."""

    constraint_id: str
    kind: str
    entity: str
    hard: bool = True
    attribute: Optional[str] = None
    op: Optional[str] = None
    value: Any = None
    payload: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""


@dataclass
class ConstraintEvaluation:
    constraint: ConstraintIR
    truth: str
    reason: str = ""
    missing: List[AcquisitionRequest] = field(default_factory=list)
    lower_bound: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    certificate: Optional[InfeasibilityCertificate] = None


@dataclass
class PropagationReport:
    truth: str
    evaluations: List[ConstraintEvaluation] = field(default_factory=list)
    missing: List[AcquisitionRequest] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def satisfied(self) -> int:
        return sum(item.truth == SAT for item in self.evaluations)

    @property
    def unresolved(self) -> int:
        return sum(item.truth == UNKNOWN for item in self.evaluations)


# =============================================================================
# 3. General helpers
# =============================================================================


def _json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


def _canonical_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _parse_hhmm(value: Any) -> Optional[int]:
    """Parse a clock time into minutes from midnight."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        if 0 <= v <= 24 * 60:
            return v
    text = str(value).strip()
    # ISO-ish datetime: keep HH:MM part.
    match = re.search(r"(?:T|\s)(\d{1,2}):(\d{2})", text)
    if not match:
        match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        if (0 <= h <= 23 and 0 <= m <= 59) or (h == 24 and m == 0):
            return h * 60 + m
    return None


def _fmt_hhmm(minutes: int) -> str:
    minutes = max(0, min(24 * 60, int(round(minutes))))
    if minutes == 24 * 60:
        return "24:00"
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _parse_duration_minutes(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Plain numeric durations from route tools are usually minutes.
        return max(0, int(round(float(value))))
    text = str(value).casefold()
    hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours|小时)", text)
    mins = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes|分钟)", text)
    total = 0.0
    if hours:
        total += float(hours.group(1)) * 60
    if mins:
        total += float(mins.group(1))
    if total > 0:
        return int(round(total))
    number = _to_float(value)
    return int(round(number)) if number is not None else None


def _parse_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def _deep_get(data: Dict[str, Any], *keys: str) -> Any:
    """Return first non-null alias found at any dictionary depth <= 3."""
    wanted = {_canonical_text(k).replace(" ", "_") for k in keys}

    def walk(obj: Any, depth: int) -> Any:
        if depth > 3:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                ck = _canonical_text(k).replace(" ", "_")
                if ck in wanted and v is not None:
                    return v
            for v in obj.values():
                found = walk(v, depth + 1)
                if found is not None:
                    return found
        return None

    return walk(data, 0)


def _extract_records(payload: Any) -> List[Dict[str, Any]]:
    """Best-effort extraction of row-like records from arbitrary tool JSON."""
    if isinstance(payload, list):
        rows: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(item)
            elif isinstance(item, list):
                # Transport tools return [[route_dict], [route_dict], ...].
                rows.extend(x for x in item if isinstance(x, dict))
        return rows
    if not isinstance(payload, dict):
        return []

    preferred = (
        "results", "data", "items", "records", "hotels", "flights", "trains",
        "attractions", "restaurants", "routes", "locations", "options",
    )
    for key in preferred:
        value = payload.get(key)
        if isinstance(value, list) and any(isinstance(x, dict) for x in value):
            return [x for x in value if isinstance(x, dict)]

    # A single record-like object.
    scalar_count = sum(
        1 for v in payload.values()
        if not isinstance(v, (dict, list))
    )
    if scalar_count >= 2:
        return [payload]

    # One nested list level.
    for value in payload.values():
        if isinstance(value, list) and any(isinstance(x, dict) for x in value):
            return [x for x in value if isinstance(x, dict)]
    return []


def _payload_has_error(payload: Any) -> bool:
    return bool(
        isinstance(payload, dict)
        and (payload.get("error") or payload.get("error_type"))
    )


def _payload_declares_empty(payload: Any) -> bool:
    """Conservatively decide whether the raw tool observation explicitly means empty.

    A parser that fails to recognize a non-empty observation must never be treated as
    a successful empty query, because that can create false infeasibility proofs.
    """
    if payload is None:
        return True
    if isinstance(payload, list):
        return len(payload) == 0
    if not isinstance(payload, dict):
        return False
    if _payload_has_error(payload):
        return False
    raw = payload.get("raw")
    if isinstance(raw, str):
        text = _canonical_text(raw)
        if not text:
            return True
        empty_markers = (
            "no results", "not found", "no matching", "nothing found",
            "未找到", "无结果", "没有找到", "暂无", "不存在",
        )
        return any(marker in text for marker in empty_markers)
    recognized_lists = (
        "results", "data", "items", "records", "hotels", "flights", "trains",
        "attractions", "restaurants", "routes", "locations", "options",
    )
    present = [payload.get(key) for key in recognized_lists if key in payload]
    if present and all(isinstance(value, list) for value in present):
        return all(len(value) == 0 for value in present)
    return False


def _stable_id(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _coord(lat: Any, lon: Any) -> Optional[str]:
    try:
        return f"{float(lat):.6f},{float(lon):.6f}"
    except (TypeError, ValueError):
        return None


def _parse_labeled_text(text: str) -> Dict[str, Any]:
    """Parse official zh/en detail-tool text without inventing any values."""
    out: Dict[str, Any] = {}
    aliases = {
        # Attraction detail fields.
        "景点id": "attraction_id", "attraction id": "attraction_id",
        "景点名称": "attraction_name", "attraction name": "attraction_name",
        "闭馆日期": "closing_dates", "closed dates": "closing_dates",
        "建议最短游玩时长": "min_visit_hours", "minimum visit duration": "min_visit_hours",
        "建议最长游玩时长": "max_visit_hours", "maximum visit duration": "max_visit_hours",
        "门票价格": "ticket_price", "ticket price": "ticket_price",
        "景点类型": "attraction_type", "attraction type": "attraction_type",

        # Restaurant detail fields.  The official zh/en tool variants may emit
        # human-readable labeled text rather than JSON, so keep these aliases in
        # the same conservative parser used for attraction details.
        "餐厅id": "restaurant_id", "restaurant id": "restaurant_id",
        "餐厅名称": "restaurant_name", "restaurant name": "restaurant_name",
        "餐馆名称": "restaurant_name",
        "人均消费": "price_per_person", "人均价格": "price_per_person",
        "平均消费": "price_per_person", "average price": "price_per_person",
        "price per person": "price_per_person", "per capita": "price_per_person",
        "菜系": "cuisine", "菜品类型": "cuisine", "cuisine": "cuisine",
        "标签": "tags", "特色标签": "tags", "tags": "tags",
        "附近地点": "nearby_location", "附近景点": "nearby_location",
        "nearby location": "nearby_location", "nearby attraction": "nearby_location",

        # Shared fields.
        "所属城市": "city", "city": "city",
        "地址": "address", "address": "address",
        "用户评分": "rating", "评分": "rating", "rating": "rating",
    }
    for line in (text or "").splitlines():
        if "：" in line:
            key, value = line.split("：", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        canonical = aliases.get(_canonical_text(key))
        if canonical:
            out[canonical] = value.strip()
        key_norm = _canonical_text(key)
        if key_norm in {"经纬度坐标", "coordinates"}:
            nums = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(nums) >= 2:
                out["latitude"], out["longitude"] = nums[0], nums[1]
        if key_norm in {"开放时间", "opening hours"}:
            times = re.findall(r"\b\d{1,2}:\d{2}\b", value)
            if times:
                out["opening_time"] = times[0]
                out["closing_time"] = times[-1]
    return out


def _parse_recommendation_text(text: str, language: str = "zh") -> List[Dict[str, Any]]:
    """Extract recommendation names/types from the official tool's prose list."""
    records: List[Dict[str, Any]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("推荐的景点", "Recommended attractions")):
            continue
        if language == "zh":
            name = line.split("，", 1)[0].strip()
            match = re.search(r"这是一个(.+?)类型的景点", line)
        else:
            name = line.split(",", 1)[0].strip()
            match = re.search(r"This is an? (.+?) type attraction", line, re.I)
        if name:
            records.append({
                "attraction_name": name,
                "attraction_type": match.group(1).strip() if match else None,
                "official_recommendation": line,
            })
    return records


def _compare(actual: Any, op: str, expected: Any) -> Optional[bool]:
    if actual is None:
        return None
    if op == "contains":
        if isinstance(actual, (list, tuple, set)):
            return any(_canonical_text(expected) in _canonical_text(item) for item in actual)
        return _canonical_text(expected) in _canonical_text(actual)
    if op == "in":
        if not isinstance(expected, list):
            return False
        return any(_canonical_text(actual) == _canonical_text(v) for v in expected)
    if op in {"==", "!="} and isinstance(actual, str):
        same = _canonical_text(actual) == _canonical_text(expected)
        return same if op == "==" else not same
    try:
        a = float(actual) if isinstance(expected, (int, float)) else actual
        if op == "==": return a == expected
        if op == "!=": return a != expected
        if op == "<": return a < expected
        if op == "<=": return a <= expected
        if op == ">": return a > expected
        if op == ">=": return a >= expected
    except (TypeError, ValueError):
        return None
    return None


# =============================================================================
# 4. Model runtime (OpenAI-compatible: OpenAI / Qwen / DeepSeek / etc.)
# =============================================================================

_MODEL_CACHE: Dict[str, Tuple[Dict[str, Any], Any]] = {}
_MODEL_LOCK = threading.Lock()


def _load_env_from_dotenv(anchor: Path) -> None:
    candidates = [anchor.parent.parent / ".env", anchor.parent.parent.parent / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return
        except Exception:
            return


def _find_model_config(anchor: Path) -> Path:
    candidates = [
        anchor.parent.parent / "models_config.json",
        anchor.parent.parent.parent / "models_config.json",
        Path.cwd() / "models_config.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("models_config.json not found")


def _load_model_config(config_name: str, anchor: Path) -> Dict[str, Any]:
    with open(_find_model_config(anchor), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    models = cfg.get("models", {})
    if config_name not in models:
        raise ValueError(f"Unknown model config {config_name!r}; available={list(models)}")
    return models[config_name]


def _model_runtime(config_name: str, anchor: Path) -> Tuple[Dict[str, Any], Any]:
    with _MODEL_LOCK:
        runtime = _MODEL_CACHE.get(config_name)
        if runtime is not None:
            return runtime
        cfg = _load_model_config(config_name, anchor)
        if cfg.get("model_type", "openai") != "openai":
            raise ValueError("This consolidated file currently supports OpenAI-compatible APIs")
        api_key_env = cfg.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        if not api_key:
            raise RuntimeError(f"Missing API key env: {api_key_env}")
        client = openai.OpenAI(api_key=api_key, base_url=cfg["base_url"])
        runtime = (cfg, client)
        _MODEL_CACHE[config_name] = runtime
        return runtime


# =============================================================================
# 5. Official tool runtime
# =============================================================================


class ToolRuntime:
    """Loads official schemas/classes and executes only official tool instances."""

    DB_MAPPING = {
        "query_train_info": "trains/trains.csv",
        "query_flight_info": "flights/flights.csv",
        "query_hotel_info": "hotels/hotels.csv",
        "query_attraction_details": "attractions/attractions.csv",
        "recommend_attractions": "attractions/attractions.csv",
        "search_location": "locations/locations_coords.csv",
        "query_road_route_info": "transportation/distance_matrix.csv",
        "recommend_restaurants": "restaurants/restaurants.csv",
        "query_restaurant_details": "restaurants/restaurants.csv",
    }

    def __init__(
        self,
        anchor: Path,
        sample_id: Optional[Any],
        database_base_path: Path,
        tool_schema_path: Path,
        language: str,
    ) -> None:
        self.anchor = anchor
        self.sample_id = sample_id
        self.database_base_path = Path(database_base_path)
        self.tool_schema_path = Path(tool_schema_path)
        self.language = language
        self.schemas = self._load_schemas()
        self.schema_by_name = {
            s["function"]["name"]: s
            for s in self.schemas
            if isinstance(s, dict)
            and isinstance(s.get("function"), dict)
            and s["function"].get("name")
        }
        self.instances = self._load_instances()

    def _load_schemas(self) -> List[Dict[str, Any]]:
        with open(self.tool_schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and isinstance(raw.get("tools"), list):
            raw = raw["tools"]
        if not isinstance(raw, list):
            raw = [raw]
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function" and isinstance(item.get("function"), dict):
                out.append(item)
            elif item.get("name"):
                out.append({
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item.get("description", ""),
                        "parameters": item.get("parameters", {}),
                    },
                })
        return out

    def _build_tool_config(self, tool_cls: Any) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"language": self.language}
        if self.sample_id is None:
            return cfg
        tool_name = getattr(tool_cls, "name", "")
        relative = self.DB_MAPPING.get(tool_name)
        if relative:
            path = self.database_base_path / f"id_{self.sample_id}" / relative
            if path.exists():
                cfg["database_path"] = str(path)
        return cfg

    def _load_instances(self) -> Dict[str, Any]:
        tools_dir = self.anchor.parent.parent / "tools"
        sys.path.insert(0, str(tools_dir.parent))
        sys.path.insert(0, str(tools_dir))
        try:
            import tools  # noqa: F401
            base_mod = importlib.import_module("tools.base_travel_tool")
            base_cls = getattr(base_mod, "BaseTravelTool")
        except Exception as e:
            raise RuntimeError(f"Could not import official travel tools: {e}") from e

        instances: Dict[str, Any] = {}
        for cls in base_cls.__subclasses__():
            try:
                inst = cls(cfg=self._build_tool_config(cls))
                name = getattr(inst, "name", None) or getattr(cls, "name", None)
                if name:
                    instances[name] = inst
            except Exception:
                continue
        return instances

    def has(self, name: str) -> bool:
        return name in self.instances and name in self.schema_by_name

    def schema(self, name: str) -> Optional[Dict[str, Any]]:
        return self.schema_by_name.get(name)

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        inst = self.instances.get(name)
        if inst is None:
            return {"error": f"tool {name!r} not available"}
        try:
            value = inst.call(arguments)
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {"raw": value}
            return value
        except Exception as e:
            return {"error_type": "TOOL_PROTOCOL_ERROR", "error": str(e)}


class OfficialToolAdapters:
    """Protocol-specific argument builders for the nine fixed official tools."""

    @staticmethod
    def build(request: AcquisitionRequest, state: Optional["TravelState"]) -> Dict[str, str]:
        hint = request.args_hint
        tool = request.tool

        def required(*names: str) -> str:
            for name in names:
                value = hint.get(name)
                if value is not None and value != "":
                    return str(value)
            raise ValueError(f"{tool} requires one of {names}; request={request.key}")

        if tool in {"query_train_info", "query_flight_info"}:
            args = {"origin": required("origin"), "destination": required("destination"),
                    "depDate": required("date", "start_date", "depDate")}
            if hint.get("seat_class"):
                args["seatClassName"] = str(hint["seat_class"])
            return args
        if tool == "query_hotel_info":
            args = {"destination": required("city", "destination"),
                    "checkinDate": required("start_date", "checkinDate"),
                    "checkoutDate": required("end_date", "checkoutDate")}
            if hint.get("hotel_star") is not None:
                star = float(hint["hotel_star"])
                args["hotelStar"] = str(int(star)) if star.is_integer() else str(star)
            if hint.get("hotel_brand"):
                args["hotelBrands"] = str(hint["hotel_brand"])
            return args
        if tool == "recommend_attractions":
            args = {"city": required("city", "destination")}
            if hint.get("attraction_type"):
                args["attraction_type"] = str(hint["attraction_type"])
            return args
        if tool == "query_attraction_details":
            return {"attraction_name": required("name", "attraction_name")}
        if tool == "query_restaurant_details":
            return {"restaurant_name": required("name", "restaurant_name")}
        if tool == "search_location":
            return {"place_name": required("name", "place_name")}
        if tool == "recommend_restaurants":
            lat = hint.get("latitude")
            lon = hint.get("longitude")
            if (lat is None or lon is None) and state is not None:
                location = state.coordinates(str(hint.get("nearby_location") or ""))
                if location:
                    lat, lon = location.split(",", 1)
            if lat is None or lon is None:
                raise ValueError(f"restaurant recommendation requires official coordinates: {request.key}")
            return {"latitude": str(lat), "longitude": str(lon)}
        if tool == "query_road_route_info":
            return {"origin": required("origin"), "destination": required("destination")}
        raise ValueError(f"unsupported official tool: {tool}")


# =============================================================================
# 6. Structured state and canonicalization
# =============================================================================


class TravelState:
    def __init__(self, problem_ir: Dict[str, Any]) -> None:
        self.problem_ir = problem_ir
        self.raw_evidence: Dict[str, Dict[str, Any]] = {}
        self.records: Dict[str, EvidenceRecord] = {}
        self.by_kind: Dict[str, List[str]] = {
            "flight": [], "train": [], "hotel": [], "attraction": [],
            "restaurant": [], "route": [], "location": [],
        }
        self.route_index: Dict[Tuple[str, str], str] = {}
        self.name_index: Dict[Tuple[str, str], str] = {}
        # queried_keys is retained for diagnostics/backward compatibility and means
        # "attempted at least once", not "successfully resolved".
        self.queried_keys: set[str] = set()
        self.resolved_keys: set[str] = set()
        self.request_attempts: Dict[str, int] = {}
        self.failed_requests: Dict[str, str] = {}
        self.query_outcomes: Dict[str, QueryOutcome] = {}
        self.solution: Optional[Dict[str, Any]] = None
        self.verification: Optional[Dict[str, Any]] = None
        self.converted_plan: Optional[Dict[str, Any]] = None

    def add_evidence(self, request: AcquisitionRequest, args: Dict[str, Any], payload: Any) -> None:
        self.queried_keys.add(request.key)
        self.request_attempts[request.key] = self.request_attempts.get(request.key, 0) + 1
        self.raw_evidence[request.key] = {
            "tool": request.tool,
            "purpose": request.purpose,
            "arguments": args,
            "payload": payload,
        }
        if isinstance(payload, dict) and payload.get("error"):
            self.failed_requests[request.key] = str(payload["error"])

    def record_query_outcome(
        self, request: AcquisitionRequest, payload: Any,
        records: Sequence[EvidenceRecord], exhaustive: bool,
    ) -> None:
        error = None
        if _payload_has_error(payload):
            if isinstance(payload, dict):
                error = str(payload.get("error") or payload.get("error_type") or "tool error")
            status = QUERY_ERROR
        elif records:
            status = QUERY_SUCCESS_NONEMPTY
        elif _payload_declares_empty(payload):
            status = QUERY_SUCCESS_EMPTY
        else:
            status = QUERY_PARSE_ERROR
            error = "non-empty tool observation produced no canonical records"
        resolved = status in {QUERY_SUCCESS_EMPTY, QUERY_SUCCESS_NONEMPTY}
        if resolved:
            self.resolved_keys.add(request.key)
            self.failed_requests.pop(request.key, None)
        else:
            self.resolved_keys.discard(request.key)
            if error:
                self.failed_requests[request.key] = error
        self.query_outcomes[request.key] = QueryOutcome(
            status=status, exhaustive=bool(exhaustive and resolved), tool=request.tool,
            record_count=len(records), error=error,
        )

    def is_resolved(self, key: str) -> bool:
        return key in self.resolved_keys

    def can_attempt(self, key: str, max_attempts: int = 2) -> bool:
        return not self.is_resolved(key) and self.request_attempts.get(key, 0) < max_attempts

    def complete_evidence_keys(self, requests: Sequence[AcquisitionRequest]) -> Optional[Tuple[str, ...]]:
        keys = tuple(request.key for request in requests)
        if not keys:
            return None
        outcomes = [self.query_outcomes.get(key) for key in keys]
        if all(outcome is not None and outcome.exhaustive
               and outcome.status in {QUERY_SUCCESS_EMPTY, QUERY_SUCCESS_NONEMPTY}
               for outcome in outcomes):
            return keys
        return None

    def add_record(self, record: EvidenceRecord) -> None:
        # Detail tools may return the same entity with a different source-specific ID.
        # Merge by canonical (kind, name) so later detail evidence enriches the
        # existing candidate instead of creating a disconnected duplicate.
        name_key = _canonical_text(record.canonical.get("name"))
        # A train number can recur on different dates/directions, and route
        # records must be indexed by their ordered endpoints, not their names.
        existing_id = self.name_index.get((record.kind, name_key)) if name_key and record.kind not in {"train", "flight", "route"} else None
        def informative_updates(values: Dict[str, Any]) -> Dict[str, Any]:
            # A detail endpoint may omit fields already returned by a recommendation
            # endpoint.  Empty lists/dicts are missing observations, not evidence that
            # an entity has no tags/services, so they must not erase grounded facts.
            return {
                key: value for key, value in values.items()
                if value is not None and value != ""
                and not (isinstance(value, (list, tuple, set, dict)) and not value)
            }
        def merge_nearby_locations(existing: EvidenceRecord, incoming: EvidenceRecord) -> None:
            if existing.kind != "restaurant":
                return
            anchors = {
                str(value).strip()
                for value in (
                    list(existing.canonical.get("nearby_locations") or [])
                    + [existing.canonical.get("nearby_location")]
                    + list(incoming.canonical.get("nearby_locations") or [])
                    + [incoming.canonical.get("nearby_location")]
                )
                if value is not None and str(value).strip()
            }
            if anchors:
                existing.canonical["nearby_locations"] = sorted(anchors)
        if existing_id and existing_id in self.records:
            existing = self.records[existing_id]
            existing.raw = {**existing.raw, **record.raw}
            merge_nearby_locations(existing, record)
            existing.canonical.update(informative_updates(record.canonical))
            record = existing
        elif record.record_id in self.records:
            existing = self.records[record.record_id]
            existing.raw = {**existing.raw, **record.raw}
            merge_nearby_locations(existing, record)
            existing.canonical.update(informative_updates(record.canonical))
            record = existing
        else:
            if record.kind == "restaurant" and record.canonical.get("nearby_location"):
                record.canonical["nearby_locations"] = [record.canonical["nearby_location"]]
            self.records[record.record_id] = record
            self.by_kind.setdefault(record.kind, []).append(record.record_id)
            if name_key:
                self.name_index[(record.kind, name_key)] = record.record_id

        if record.kind == "route":
            origin = _canonical_text(record.canonical.get("origin"))
            dest = _canonical_text(record.canonical.get("destination"))
            if origin and dest:
                self.route_index[(origin, dest)] = record.record_id

    def records_of(self, kind: str) -> List[EvidenceRecord]:
        return [self.records[rid] for rid in self.by_kind.get(kind, [])]

    def route(self, origin: str, destination: str) -> Optional[EvidenceRecord]:
        rid = self.route_index.get((_canonical_text(origin), _canonical_text(destination)))
        return self.records.get(rid) if rid else None

    def entity(self, name: str) -> Optional[EvidenceRecord]:
        key = _canonical_text(name)
        for kind in ("location", "hotel", "attraction", "restaurant"):
            rid = self.name_index.get((kind, key))
            if rid and rid in self.records:
                return self.records[rid]
        return None

    def coordinates(self, name: str) -> Optional[str]:
        rec = self.entity(name)
        if rec is None:
            return None
        return _coord(rec.canonical.get("lat"), rec.canonical.get("lon"))

    def compact_summary(self) -> Dict[str, Any]:
        return {
            "counts": {kind: len(ids) for kind, ids in self.by_kind.items()},
            "queried_keys": sorted(self.queried_keys),
            "resolved_keys": sorted(self.resolved_keys),
            "request_attempts": dict(self.request_attempts),
            "failed_requests": self.failed_requests,
            "query_outcomes": {
                key: {
                    "status": outcome.status, "exhaustive": outcome.exhaustive,
                    "tool": outcome.tool, "record_count": outcome.record_count,
                }
                for key, outcome in self.query_outcomes.items()
            },
            "sample_records": {
                kind: [self.records[r].canonical for r in ids[:3]]
                for kind, ids in self.by_kind.items() if ids
            },
        }


class Canonicalizer:
    """Maps tool rows to a tolerant canonical schema without hidden-data access."""

    @staticmethod
    def _record_id(tool: str, kind: str, raw: Dict[str, Any], idx: int) -> str:
        explicit = _deep_get(raw, "id", "record_id", "product_id", "hotel_id", "attraction_id",
                             "restaurant_id", "flight_no", "flight_number", "train_no", "train_number")
        if explicit is not None:
            return f"{kind}:{explicit}"
        return f"{kind}:{tool}:{idx}:{uuid.uuid5(uuid.NAMESPACE_OID, json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)).hex[:12]}"

    @staticmethod
    def _name(raw: Dict[str, Any], kind: str) -> str:
        aliases = {
            "hotel": ("hotel_name", "name", "title"),
            "attraction": ("attraction_name", "name", "title"),
            "restaurant": ("restaurant_name", "name", "title"),
            "flight": ("flight_no", "flight_number", "flight", "name"),
            "train": ("train_no", "train_number", "train", "name"),
            "location": ("place_name", "name", "location_name", "address"),
            "route": ("route_name", "name"),
        }
        return str(_deep_get(raw, *aliases.get(kind, ("name", "title"))) or "").strip()

    @classmethod
    def canonicalize(cls, tool: str, payload: Any, language: str = "zh") -> List[EvidenceRecord]:
        kind = {
            "query_flight_info": "flight",
            "query_train_info": "train",
            "query_hotel_info": "hotel",
            "recommend_attractions": "attraction",
            "query_attraction_details": "attraction",
            "recommend_restaurants": "restaurant",
            "query_restaurant_details": "restaurant",
            "query_road_route_info": "route",
            "search_location": "location",
        }.get(tool)
        if not kind or _payload_has_error(payload):
            return []

        # Several official tools intentionally return human-readable text. Parse
        # only values present in that observation; no database is accessed here.
        if isinstance(payload, dict) and isinstance(payload.get("raw"), str):
            text = payload["raw"]
            if tool == "recommend_attractions":
                rows = _parse_recommendation_text(text, language)
            elif tool == "query_attraction_details":
                parsed = _parse_labeled_text(text)
                rows = [parsed] if parsed.get("attraction_name") else []
            elif tool == "query_restaurant_details":
                parsed = _parse_labeled_text(text)
                rows = [parsed] if parsed.get("restaurant_name") else []
            else:
                rows = []
        else:
            rows = _extract_records(payload)
        records: List[EvidenceRecord] = []
        for idx, raw in enumerate(rows):
            # One transport candidate contains one or more numbered segments.
            # Preserve the entire route and derive endpoints/times from first/last.
            if kind in {"flight", "train"}:
                segments = [
                    value for key, value in raw.items()
                    if isinstance(value, dict) and re.search(r"(?:segment|第)\s*\d+", str(key), re.I)
                ]
                if segments:
                    first, last = segments[0], segments[-1]
                    numbers = [str(_deep_get(s, "marketingTransportNo") or "") for s in segments]
                    name = "+".join(x for x in numbers if x)
                    canonical = {
                        "name": name,
                        "price": _to_float(raw.get("price")),
                        "rating": None,
                        "city": None,
                        "address": None,
                        "opening_time": None,
                        "closing_time": None,
                        "mode": kind,
                        "origin": _deep_get(first, "depCityName"),
                        "destination": _deep_get(last, "arrCityName"),
                        "departure_station": _deep_get(first, "depStationName"),
                        "arrival_station": _deep_get(last, "arrStationName"),
                        "departure_time": _parse_hhmm(_deep_get(first, "depDateTime")),
                        "arrival_time": _parse_hhmm(_deep_get(last, "arrDateTime")),
                        "departure_date": str(_deep_get(first, "depDateTime") or "")[:10],
                        "arrival_date": str(_deep_get(last, "arrDateTime") or "")[:10],
                        "duration": sum(int(_to_float(_deep_get(s, "duration")) or 0) for s in segments),
                        "direct": len(segments) == 1,
                        "seat_class": _deep_get(first, "seatClassName"),
                        "train_type": _deep_get(first, "marketingTransportName"),
                        "seat_status": min(
                            (_to_float(_deep_get(s, "剩余票数量", "Remaining Seats")) or float("inf"))
                            for s in segments
                        ),
                        "segments": segments,
                    }
                    rid = f"{kind}:{name or 'unknown'}:{_stable_id(raw)}"
                    records.append(EvidenceRecord(rid, kind, tool, raw, canonical))
                    continue

            name = cls._name(raw, kind)
            canonical: Dict[str, Any] = {
                "name": name,
                "price": _to_float(_deep_get(raw, "price", "cost", "ticket_price", "fare", "average_price", "avg_price", "per_capita", "price_per_person", "room_price")),
                "rating": _to_float(_deep_get(raw, "rating", "score", "rating_score", "average_score")),
                "city": _deep_get(raw, "city", "city_name"),
                "address": _deep_get(raw, "address", "location", "location_name"),
                "opening_time": _parse_hhmm(_deep_get(raw, "opening_time", "open_time", "opening", "open")),
                "closing_time": _parse_hhmm(_deep_get(raw, "closing_time", "close_time", "closing", "close")),
                "lat": _to_float(_deep_get(raw, "lat", "latitude")),
                "lon": _to_float(_deep_get(raw, "lon", "lng", "longitude")),
            }

            if kind == "hotel":
                canonical.update({
                    "star": _to_float(_deep_get(raw, "hotelStar", "hotel_star", "star")),
                    "services": _deep_get(raw, "services", "service", "amenities") or [],
                    "brand": _deep_get(raw, "brand", "hotel_brand"),
                    "decoration_year": _to_float(_deep_get(raw, "decorationTime", "decoration_year", "renovation_year")),
                })

            if kind in {"attraction", "restaurant"}:
                min_hours = _to_float(_deep_get(raw, "min_visit_hours", "min_duration_hours", "minimum_visit_hours"))
                max_hours = _to_float(_deep_get(raw, "max_visit_hours", "max_duration_hours", "maximum_visit_hours"))
                canonical["min_duration"] = int(round(min_hours * 60)) if min_hours is not None else None
                canonical["max_duration"] = int(round(max_hours * 60)) if max_hours is not None else None
                canonical["cuisine"] = _deep_get(raw, "cuisine", "cuisine_type", "category", "type")
                canonical["tags"] = _deep_get(raw, "tags", "services") or []
                canonical["nearby_location"] = _deep_get(raw, "nearby_attraction_name", "nearby_location")
                canonical["attraction_type"] = _deep_get(raw, "attraction_type", "type")
                if kind == "attraction":
                    canonical["closing_dates"] = _deep_get(raw, "closing_dates", "closed_dates", "closed_days")

            if kind in {"flight", "train"}:
                canonical.update({
                    "mode": kind,
                    "origin": _deep_get(raw, "origin", "departure_city", "from_city", "departure"),
                    "destination": _deep_get(raw, "destination", "arrival_city", "to_city", "arrival"),
                    "departure_station": _deep_get(raw, "departure_station", "departure_airport", "from_station", "origin_station"),
                    "arrival_station": _deep_get(raw, "arrival_station", "arrival_airport", "to_station", "destination_station"),
                    "departure_time": _parse_hhmm(_deep_get(raw, "departure_time", "depart_time", "start_time")),
                    "arrival_time": _parse_hhmm(_deep_get(raw, "arrival_time", "arrive_time", "end_time")),
                    "duration": _parse_duration_minutes(_deep_get(raw, "duration", "travel_time", "elapsed_time")),
                    "direct": _deep_get(raw, "direct", "is_direct", "nonstop", "non_stop"),
                })
                if canonical["duration"] is None and canonical["departure_time"] is not None and canonical["arrival_time"] is not None:
                    delta = canonical["arrival_time"] - canonical["departure_time"]
                    if delta < 0:
                        delta += 24 * 60
                    canonical["duration"] = delta

            if kind == "route":
                canonical.update({
                    "origin": _deep_get(raw, "origin", "from", "from_location", "start", "start_location"),
                    "destination": _deep_get(raw, "destination", "to", "to_location", "end", "end_location"),
                    "distance": _deep_get(raw, "distance", "distance_km", "distance_in_meters"),
                    "duration": _parse_duration_minutes(_deep_get(raw, "duration", "travel_time", "time", "minutes", "duration_in_minutes")),
                    "mode": _deep_get(raw, "mode", "transport_mode", "travel_mode") or "taxi",
                })
                # Route tools sometimes omit explicit origin/destination from the returned row.
                if not canonical["origin"] and isinstance(payload, dict):
                    canonical["origin"] = _deep_get(payload, "origin", "from", "from_location")
                if not canonical["destination"] and isinstance(payload, dict):
                    canonical["destination"] = _deep_get(payload, "destination", "to", "to_location")

            rid = cls._record_id(tool, kind, raw, idx)
            records.append(EvidenceRecord(rid, kind, tool, raw, canonical))
        return records


# =============================================================================
# 7. Incremental symbolic constraint layer
# =============================================================================


class ConstraintCompiler:
    """Compile normalized ProblemIR into a small typed constraint registry."""

    @staticmethod
    def compile(problem_ir: Dict[str, Any]) -> List[ConstraintIR]:
        constraints: List[ConstraintIR] = []

        def add_predicates(entity: str, spec: Dict[str, Any], prefix: str) -> None:
            for index, predicate in enumerate(spec.get("constraints") or []):
                constraints.append(ConstraintIR(
                    f"{prefix}.hard.{index}", "predicate", entity, True,
                    attribute=str(predicate.get("field") or ""),
                    op=str(predicate.get("op") or "=="), value=predicate.get("value"),
                    source_text=str(predicate.get("source_text") or ""),
                ))
            for index, predicate in enumerate(spec.get("soft_preferences") or []):
                constraints.append(ConstraintIR(
                    f"{prefix}.soft.{index}", "predicate", entity, False,
                    attribute=str(predicate.get("field") or ""),
                    op=str(predicate.get("op") or "=="), value=predicate.get("value"),
                    source_text=str(predicate.get("source_text") or ""),
                ))
            objective = spec.get("objective")
            if isinstance(objective, dict):
                constraints.append(ConstraintIR(
                    f"{prefix}.objective", "arg_extreme", entity,
                    not bool(objective.get("soft")), attribute=str(objective.get("field") or ""),
                    op=str(objective.get("sense") or "min"),
                    payload={"where": list(spec.get("constraints") or [])},
                    source_text=str(objective.get("source_text") or ""),
                ))

        for direction in ("outbound", "return"):
            add_predicates("transport", problem_ir["transport"][direction], f"transport.{direction}")
            constraints.append(ConstraintIR(
                f"transport.{direction}.coverage", "candidate_coverage", "transport", True,
                payload={"direction": direction},
            ))
        add_predicates("hotel", problem_ir["hotel"], "hotel")
        constraints.append(ConstraintIR("hotel.coverage", "candidate_coverage", "hotel", True))
        add_predicates("attraction", problem_ir["attractions"], "attractions")
        constraints.append(ConstraintIR(
            "attractions.coverage", "attraction_coverage", "attraction", True,
            payload={
                "selection": problem_ir["attractions"].get("selection", "default"),
                "count": problem_ir["attractions"].get("count"),
                "required_names": list(problem_ir["attractions"].get("required_names") or []),
                "constraints": list(problem_ir["attractions"].get("constraints") or []),
                "selection_groups": list(problem_ir["attractions"].get("selection_groups") or []),
                "required_groups": list(problem_ir["attractions"].get("required_groups") or []),
            },
        ))
        for index, requirement in enumerate((problem_ir.get("restaurants") or {}).get("requirements") or []):
            constraints.append(ConstraintIR(
                f"restaurants.requirement.{index}", "restaurant_requirement", "restaurant", True,
                payload={**requirement, "requirement_index": index},
            ))
        budget = problem_ir["trip"].get("budget_max")
        if budget is not None:
            constraints.append(ConstraintIR(
                "trip.budget", "budget_lower_bound", "trip", True,
                op="<=", value=float(budget),
            ))
        return constraints


class IncrementalConstraintEngine:
    """Evaluate typed constraints over partial official-tool evidence."""

    def __init__(self, planner: "SolverInformedTravelAgent") -> None:
        self.planner = planner
        self.constraints = ConstraintCompiler.compile(planner.problem_ir)

    def _result(
        self, constraint: ConstraintIR, truth: str, reason: str = "",
        missing: Optional[List[AcquisitionRequest]] = None,
        lower_bound: Optional[float] = None,
        certificate: Optional[InfeasibilityCertificate] = None,
        **diagnostics: Any,
    ) -> ConstraintEvaluation:
        return ConstraintEvaluation(
            constraint, truth, reason, missing or [], lower_bound, diagnostics, certificate,
        )

    def _domain_requests(
        self, entity: str, direction: Optional[str] = None,
    ) -> List[AcquisitionRequest]:
        requests = self.planner._initial_requests()
        if entity == "transport" and direction:
            return [request for request in requests if request.key.startswith(f"{direction}:")]
        if entity == "hotel":
            return [request for request in requests if request.tool == "query_hotel_info"]
        return []

    def _complete_domain(
        self, entity: str, direction: Optional[str] = None,
    ) -> Optional[Tuple[str, ...]]:
        state = self.planner.state
        assert state is not None
        return state.complete_evidence_keys(self._domain_requests(entity, direction))

    @staticmethod
    def _no_witness_certificate(
        constraint: ConstraintIR, evidence_keys: Tuple[str, ...], domain: str,
        explanation: str,
    ) -> InfeasibilityCertificate:
        return InfeasibilityCertificate(
            kind="EXHAUSTIVE_NO_WITNESS",
            constraints=(constraint.constraint_id,), evidence_keys=evidence_keys,
            complete_domains=(domain,), explanation=explanation,
        )

    def _candidate_coverage(self, constraint: ConstraintIR) -> ConstraintEvaluation:
        planner, state = self.planner, self.planner.state
        assert state is not None
        if constraint.entity == "hotel":
            candidates = [
                record for record in planner._rank_records("hotel")
                if planner._status_against(record, planner.problem_ir["hotel"].get("constraints") or []) == SAT
            ]
            if candidates:
                return self._result(constraint, SAT, count=len(candidates))
            requests = self._domain_requests("hotel")
            evidence_keys = self._complete_domain("hotel")
            if evidence_keys is None:
                missing = [request for request in requests if state.can_attempt(request.key)]
                return self._result(
                    constraint, UNKNOWN, "hotel domain is incomplete or contains query errors", missing,
                )
            certificate = self._no_witness_certificate(
                constraint, evidence_keys, "hotel", "exhaustive hotel query has no hard-valid candidate",
            )
            return self._result(
                constraint, UNSAT, "no hotel satisfies hard constraints", certificate=certificate,
            )

        direction = str(constraint.payload["direction"])
        candidates: List[EvidenceRecord] = []
        for mode in planner.problem_ir["transport"][direction]["allowed_modes"]:
            candidates.extend(planner._rank_records(mode, direction))
        if candidates:
            return self._result(constraint, SAT, count=len(candidates))
        requests = self._domain_requests("transport", direction)
        evidence_keys = self._complete_domain("transport", direction)
        if evidence_keys is None:
            missing = [request for request in requests if state.can_attempt(request.key)]
            return self._result(
                constraint, UNKNOWN, f"{direction} transport domain incomplete or errored", missing,
            )
        certificate = self._no_witness_certificate(
            constraint, evidence_keys, f"transport.{direction}",
            f"all exhaustive {direction} transport queries have no hard-valid candidate",
        )
        return self._result(
            constraint, UNSAT, f"no {direction} transport satisfies hard constraints",
            certificate=certificate,
        )

    def _attraction_coverage(self, constraint: ConstraintIR) -> ConstraintEvaluation:
        planner, state = self.planner, self.planner.state
        assert state is not None
        spec = constraint.payload
        records = state.records_of("attraction")
        names = {_canonical_text(record.canonical.get("name")) for record in records}
        missing_names = [name for name in spec.get("required_names") or [] if _canonical_text(name) not in names]
        if missing_names:
            detail_requests = {
                request.key: request for request in planner._detail_requests()
                if request.tool == "query_attraction_details"
            }
            missing_requests: List[AcquisitionRequest] = []
            absent_keys: List[str] = []
            absent_names: List[str] = []
            uncertain_names: List[str] = []
            for name in missing_names:
                key = f"details:attraction:{_canonical_text(name)}"
                outcome = state.query_outcomes.get(key)
                if outcome is None:
                    request = detail_requests.get(key)
                    if request is not None:
                        missing_requests.append(request)
                    else:
                        uncertain_names.append(name)
                elif outcome.exhaustive and outcome.status == QUERY_SUCCESS_EMPTY:
                    absent_keys.append(key)
                    absent_names.append(name)
                else:
                    uncertain_names.append(name)
            if absent_keys:
                certificate = InfeasibilityCertificate(
                    kind="EXACT_ENTITY_ABSENCE",
                    constraints=(constraint.constraint_id,),
                    evidence_keys=tuple(absent_keys),
                    complete_domains=tuple(f"attraction:{name}" for name in absent_names),
                    explanation="exact attraction detail lookup returned empty",
                )
                return self._result(
                    constraint, UNSAT, "required attraction is absent by exact lookup",
                    certificate=certificate, missing_names=missing_names,
                )
            return self._result(
                constraint, UNKNOWN, "required attraction identity is not certified",
                missing_requests, missing_names=missing_names, uncertain_names=uncertain_names,
            )

        required_count = len(spec.get("required_names") or [])
        selection = spec.get("selection")
        if selection == "top_k":
            eligible = [r for r in records if planner._status_against(r, spec.get("constraints") or []) == SAT]
            required_count = max(required_count, int(spec.get("count") or 1))
            if len(eligible) < int(spec.get("count") or 1):
                return self._result(constraint, UNKNOWN, "retrieved top-k attraction domain too small",
                                    eligible=len(eligible), required=int(spec.get("count") or 1))
        elif selection == "all_matching":
            groups = spec.get("selection_groups") or []
            eligible = [r for r in records if (any(planner._status_against(r, group) == SAT for group in groups)
                                                if groups else True)]
            required_count = max(required_count, len(eligible))
        for group in spec.get("required_groups") or []:
            eligible = [r for r in records if planner._status_against(r, group.get("constraints") or []) == SAT]
            count = int(group.get("count") or 1)
            if len(eligible) < count:
                return self._result(constraint, UNKNOWN, "required attraction group coverage incomplete",
                                    eligible=len(eligible), required=count, group=group)
            required_count = max(required_count, count)

        capacity = 3 * planner._trip_days()
        if required_count > capacity:
            return self._result(
                constraint, UNKNOWN, "required attractions exceed current template capacity",
                required=required_count, capacity=capacity, model_limit=True,
            )
        return self._result(constraint, SAT, required=required_count, capacity=capacity)

    def _restaurant_requirement(self, constraint: ConstraintIR) -> ConstraintEvaluation:
        planner, state = self.planner, self.planner.state
        assert state is not None
        requirement = constraint.payload
        objective = requirement.get("objective")
        nearby = requirement.get("nearby_location")
        if (isinstance(objective, dict)
                and str(objective.get("field") or "") in {"distance", "route_distance", "travel_distance"}
                and nearby):
            base_requirement = dict(requirement)
            base_requirement["objective"] = None
            domain = planner._restaurants_for_requirement(base_requirement)
            missing_routes: Dict[str, AcquisitionRequest] = {}
            for record in domain:
                for request in planner._requests_for_route(
                        str(nearby), str(record.canonical.get("name")),
                        f"certify restaurant argmin for {constraint.constraint_id}"):
                    missing_routes[request.key] = request
            if missing_routes:
                return self._result(constraint, UNKNOWN, "argmin route facts incomplete",
                                    list(missing_routes.values()), domain_size=len(domain))
            grounded = [record for record in domain
                        if state.route(str(nearby), str(record.canonical.get("name"))) is not None]
            if domain and not grounded:
                return self._result(constraint, UNKNOWN, "argmin route domain is not certifiably complete",
                                    domain_size=len(domain))
        options = planner._restaurants_for_requirement(requirement)
        if options:
            return self._result(constraint, SAT, count=len(options))
        missing = planner._restaurant_requests()
        missing = [request for request in missing if state.can_attempt(request.key)]
        if missing:
            return self._result(constraint, UNKNOWN, "restaurant requirement needs grounding", missing)
        return self._result(constraint, UNKNOWN, "restaurant recall policy exhausted without a witness",
                            requirement_index=requirement.get("requirement_index"))

    def _budget_lower_bound(self, constraint: ConstraintIR) -> ConstraintEvaluation:
        planner = self.planner
        trip = planner.problem_ir["trip"]
        travelers, rooms, nights = int(trip["travelers"]), int(trip["rooms"]), planner._trip_days() - 1
        directions = []
        evidence_keys: List[str] = []
        for direction in ("outbound", "return"):
            complete = self._complete_domain("transport", direction)
            if complete is None:
                return self._result(constraint, UNKNOWN, f"{direction} domain not certificate-complete")
            evidence_keys.extend(complete)
            rows = []
            for mode in planner.problem_ir["transport"][direction]["allowed_modes"]:
                rows.extend(planner._rank_records(mode, direction))
            prices = [float(row.canonical["price"]) for row in rows if row.canonical.get("price") is not None]
            if not prices:
                return self._result(constraint, UNKNOWN, f"{direction} price domain incomplete")
            directions.append(min(prices) * travelers)
        hotel_complete = self._complete_domain("hotel")
        if hotel_complete is None:
            return self._result(constraint, UNKNOWN, "hotel domain not certificate-complete")
        evidence_keys.extend(hotel_complete)
        hotels = [row for row in planner._rank_records("hotel")
                  if planner._status_against(row, planner.problem_ir["hotel"].get("constraints") or []) == SAT
                  and row.canonical.get("price") is not None]
        if not hotels:
            return self._result(constraint, UNKNOWN, "hotel price domain incomplete")
        lower_bound = sum(directions) + min(float(row.canonical["price"]) for row in hotels) * rooms * nights
        truth = UNSAT if lower_bound > float(constraint.value) else UNKNOWN
        certificate = None
        if truth == UNSAT:
            certificate = InfeasibilityCertificate(
                kind="BUDGET_LOWER_BOUND",
                constraints=(constraint.constraint_id,),
                evidence_keys=tuple(evidence_keys),
                complete_domains=("transport.outbound", "transport.return", "hotel"),
                lhs=lower_bound, rhs=float(constraint.value),
                explanation="global minima over exhaustive fixed-cost domains exceed budget",
            )
        return self._result(
            constraint, truth, "fixed-cost lower bound", lower_bound=lower_bound,
            certificate=certificate, budget=float(constraint.value),
        )

    def _predicate(self, constraint: ConstraintIR) -> ConstraintEvaluation:
        planner, state = self.planner, self.planner.state
        assert state is not None
        predicate = {"field": constraint.attribute, "op": constraint.op, "value": constraint.value}
        if constraint.entity == "transport":
            direction = "outbound" if ".outbound." in constraint.constraint_id else "return"
            records: List[EvidenceRecord] = []
            for mode in planner.problem_ir["transport"][direction]["allowed_modes"]:
                records.extend(planner._rank_records(mode, direction))
        else:
            records = state.records_of(constraint.entity)
        statuses = [planner._status_against(record, [predicate]) for record in records]
        if SAT in statuses:
            return self._result(constraint, SAT, "predicate domain has a witness")
        if UNKNOWN in statuses:
            return self._result(constraint, UNKNOWN, "predicate facts incomplete")
        evidence_keys = self._complete_domain(
            constraint.entity, direction if constraint.entity == "transport" else None,
        )
        if evidence_keys is None:
            return self._result(constraint, UNKNOWN, "predicate domain is not certificate-complete")
        domain = f"transport.{direction}" if constraint.entity == "transport" else constraint.entity
        certificate = self._no_witness_certificate(
            constraint, evidence_keys, domain,
            "exhaustive domain contains no candidate satisfying the hard predicate",
        )
        return self._result(
            constraint, UNSAT, "predicate domain has no witness", certificate=certificate,
        )

    def _arg_extreme(self, constraint: ConstraintIR) -> ConstraintEvaluation:
        planner, state = self.planner, self.planner.state
        assert state is not None
        if constraint.entity == "transport":
            direction = "outbound" if ".outbound." in constraint.constraint_id else "return"
            records: List[EvidenceRecord] = []
            for mode in planner.problem_ir["transport"][direction]["allowed_modes"]:
                records.extend(planner._rank_records(mode, direction))
        else:
            records = state.records_of(constraint.entity)
        where = constraint.payload.get("where") or []
        domain = [record for record in records if planner._status_against(record, where) == SAT]
        values = [_to_float(planner._record_field(record, str(constraint.attribute))) for record in domain]
        known = [value for value in values if value is not None]
        direction = ("outbound" if ".outbound." in constraint.constraint_id else "return") \
            if constraint.entity == "transport" else None
        complete = self._complete_domain(constraint.entity, direction)
        if complete is None:
            return self._result(
                constraint, UNKNOWN, "arg-extreme domain is not certificate-complete",
                domain_size=len(domain), known_values=len(known),
            )
        if domain and len(known) == len(domain):
            optimum = max(known) if constraint.op == "max" else min(known)
            return self._result(constraint, SAT, "arg-extreme domain is complete and grounded",
                                optimum=optimum, domain_size=len(domain))
        return self._result(constraint, UNKNOWN, "arg-extreme values are not fully grounded",
                            domain_size=len(domain), known_values=len(known))

    def _evaluate(self, constraint: ConstraintIR) -> ConstraintEvaluation:
        if constraint.kind == "predicate":
            return self._predicate(constraint)
        if constraint.kind == "arg_extreme":
            return self._arg_extreme(constraint)
        if constraint.kind == "candidate_coverage":
            return self._candidate_coverage(constraint)
        if constraint.kind == "attraction_coverage":
            return self._attraction_coverage(constraint)
        if constraint.kind == "restaurant_requirement":
            return self._restaurant_requirement(constraint)
        if constraint.kind == "budget_lower_bound":
            return self._budget_lower_bound(constraint)
        return self._result(constraint, UNKNOWN, "awaiting candidate or itinerary state")

    def propagate(self) -> PropagationReport:
        evaluations = [self._evaluate(constraint) for constraint in self.constraints]
        hard_false = [item for item in evaluations if item.constraint.hard and item.truth == UNSAT]
        hard_unknown = [item for item in evaluations if item.constraint.hard and item.truth == UNKNOWN]
        missing = {
            request.key: request
            for item in evaluations if item.truth == UNKNOWN
            for request in item.missing
        }
        conflicts = [{
            "constraint_id": item.constraint.constraint_id,
            "kind": item.constraint.kind,
            "reason": item.reason,
            **item.diagnostics,
        } for item in hard_false]
        truth = UNSAT if hard_false else UNKNOWN if hard_unknown else SAT
        return PropagationReport(truth, evaluations, list(missing.values()), conflicts)


# =============================================================================
# 7. Solver-informed agent
# =============================================================================


class SolverInformedTravelAgent:
    """One-file solver-informed travel agent with explicit solve/acquire loop."""

    def __init__(
        self,
        model: str,
        sample_id: Optional[Any] = None,
        database_base_path: Optional[Path] = None,
        tool_schema_path: Optional[Path] = None,
        language: str = "zh",
        max_llm_calls: int = 100,
        max_tool_calls: int = 400,
        max_solver_rounds: int = 24,
        max_reflections: int = 2,
        max_relaxation_rounds: int = 12,
        max_relaxed_candidates: int = 8,
        enable_semantic_audit: bool = True,
        enable_reflection: bool = True,
        enable_relaxation: bool = True,
    ) -> None:
        self.anchor = Path(__file__).resolve()
        _load_env_from_dotenv(self.anchor)
        self.model = model
        self.sample_id = sample_id
        self.language = language
        self.max_llm_calls = max(1, int(max_llm_calls))
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.max_solver_rounds = max(1, int(max_solver_rounds))
        self.max_reflections = max(0, int(max_reflections))
        self.max_relaxation_rounds = max(1, int(max_relaxation_rounds))
        self.max_relaxed_candidates = max(1, int(max_relaxed_candidates))
        self.enable_semantic_audit = enable_semantic_audit
        self.enable_reflection = enable_reflection
        self.enable_relaxation = enable_relaxation

        domain_root = self.anchor.parent.parent
        self.database_base_path = Path(database_base_path or (domain_root / "database" / f"database_{language}"))
        self.tool_schema_path = Path(tool_schema_path or (domain_root / "tools" / f"tool_schema_{language}.json"))
        self.tools = ToolRuntime(
            self.anchor, sample_id, self.database_base_path,
            self.tool_schema_path, language,
        )

        self.problem_ir: Dict[str, Any] = {}
        self.state: Optional[TravelState] = None
        self.metrics: Dict[str, Any] = {}
        self._assignment_nogoods: Dict[Tuple[str, str, str], set[Tuple[str, ...]]] = {}
        self._day_nogoods: Dict[Tuple[str, str, str], set[Tuple[int, Tuple[str, ...]]]] = {}
        self.constraint_engine: Optional[IncrementalConstraintEngine] = None
        self.last_propagation: Optional[PropagationReport] = None
        self.semantic_events: List[Dict[str, Any]] = []
        self._reflection_signatures: set[str] = set()
        # Non-null only while the deterministic feasibility-restoration phase is
        # solving a softened search model.  The original IR remains the authority
        # for final verification and violation reporting.
        self._relaxation_original_ir: Optional[Dict[str, Any]] = None
        self._relaxation_profile: Optional[Dict[str, Any]] = None
        self.relaxation_report: Optional[Dict[str, Any]] = None
        self._reset_metrics()

    # ---------------------------------------------------------------------
    # Runtime accounting
    # ---------------------------------------------------------------------

    def _reset_metrics(self) -> None:
        self.metrics = {
            "llm_calls": 0,
            "api_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "tool_calls": 0,
            "tool_calls_by_name": {},
            "tool_errors": 0,
            "tool_parse_errors": 0,
            "tool_retries": 0,
            "tool_retry_exhausted": 0,
            "duplicate_tool_calls_avoided": 0,
            "solver_rounds": 0,
            "master_assignments": 0,
            "master_day_cuts": 0,
            "constraint_propagations": 0,
            "constraint_false": 0,
            "constraint_unknown": 0,
            "fastcheck_fact_requests": 0,
            "frontier_states": 0,
            "shared_fact_requests": 0,
            "relax_and_complete_rounds": 0,
            "relax_and_complete_requests": 0,
            "solver_runtime": 0.0,
            "solver_backend": "gurobi" if HAS_GUROBI else "unavailable",
            "route_queries": 0,
            "parser_success": False,
            "parser_attempts": 0,
            "parser_validation_errors": [],
            "semantic_audit_calls": 0,
            "semantic_audit_retries": 0,
            "semantic_audit_repairs": 0,
            "semantic_audit_rejected": 0,
            "recovery_rejected": 0,
            "reflection_calls": 0,
            "reflection_actions_accepted": 0,
            "reflection_duplicate_actions": 0,
            "reflection_recovery_success": False,
            "incumbent_updates": 0,
            "best_effort_returned": False,
            "relaxation_attempted": False,
            "relaxation_profiles_attempted": 0,
            "relaxation_solver_rounds": 0,
            "relaxation_candidates": 0,
            "relaxation_exact_recovered": False,
            "relaxed_feasible_returned": False,
            "relaxation_hard_violations": 0,
            "infeasibility_certificates_issued": 0,
            "infeasibility_certificates_rejected": 0,
            "verification_success": False,
        }

    def _call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
        if self.metrics["llm_calls"] >= self.max_llm_calls:
            raise RuntimeError("LLM call budget exhausted")
        cfg, client = _model_runtime(self.model, self.anchor)
        actual_model = cfg.get("model_name", self.model)
        temperature = cfg.get("temperature")
        extra_body = cfg.get("extra_body")
        max_retries = max(1, int(cfg.get("max_retries", 5)))
        backoff = float(cfg.get("backoff", 1.5))
        is_reasoning = any(x in actual_model.lower() for x in ("o1", "o3", "o4-mini", "reasoner"))

        last_error = None
        for attempt in range(max_retries):
            self.metrics["api_attempts"] += 1
            try:
                params: Dict[str, Any] = {"model": actual_model, "messages": messages}
                if tools:
                    params["tools"] = tools
                if not is_reasoning and temperature is not None:
                    params["temperature"] = temperature
                if extra_body:
                    params["extra_body"] = extra_body
                response = client.chat.completions.create(**params)
                self.metrics["llm_calls"] += 1
                usage = getattr(response, "usage", None)
                if usage:
                    inp = getattr(usage, "prompt_tokens", 0) or 0
                    out = getattr(usage, "completion_tokens", 0) or 0
                    self.metrics["input_tokens"] += int(inp)
                    self.metrics["output_tokens"] += int(out)
                    self.metrics["total_tokens"] += int(inp) + int(out)
                return response
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(30.0, backoff * (2 ** attempt)))
        raise last_error or RuntimeError("LLM call failed")

    # ---------------------------------------------------------------------
    # Problem parsing
    # ---------------------------------------------------------------------

    def _parse_problem_ir(self, user_query: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": PROBLEM_IR_PROMPT},
            {"role": "user", "content": user_query},
        ]
        last_error: Optional[Exception] = None
        for _ in range(3):
            response = self._call_llm(messages)
            self.metrics["parser_attempts"] += 1
            content = response.choices[0].message.content or ""
            try:
                ir = _json_object(content)
                self._validate_problem_ir(ir)
                self._validate_constraint_modalities(ir, user_query)
                self.metrics["parser_success"] = True
                return ir
            except Exception as e:
                last_error = e
                self.metrics["parser_validation_errors"].append(str(e))
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Validation error: {e}. Return the full corrected JSON only."},
                ]
        raise ValueError(f"ProblemIR parse failed: {last_error}")

    @staticmethod
    def _validate_problem_ir(ir: Dict[str, Any]) -> None:
        trip = ir.get("trip")
        if not isinstance(trip, dict):
            raise ValueError("trip must be an object")
        for key in ("travelers", "rooms"):
            value = trip.get(key, 1)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"trip.{key} must be a positive integer")
            trip[key] = value
        budget = trip.get("budget_max")
        if budget is not None:
            trip["budget_max"] = float(budget)
        for key in ("start_date", "end_date"):
            parsed = _parse_date(trip.get(key))
            if parsed is None:
                raise ValueError(f"trip.{key} must be a valid date")
            trip[key] = parsed.date().isoformat()

        allowed_operators = {"==", "!=", "<", "<=", ">", ">=", "contains", "in"}
        allowed_fields = {
            "transport": {
                "seat_class", "train_type", "manufacturer", "direct", "price", "duration",
                "departure_time", "arrival_time",
            },
            "hotel": {"star", "services", "brand", "decoration_year", "price", "rating"},
            "attraction": {"name", "attraction_type", "price", "rating"},
            "restaurant": {
                "name", "tags", "cuisine", "price", "rating", "nearby_location",
                "distance", "route_distance", "travel_distance",
            },
        }

        def normalize_constraints(spec: Dict[str, Any], entity: str) -> None:
            if not isinstance(spec, dict):
                raise ValueError(f"constraint section must be an object, got {type(spec).__name__}")
            raw_soft = spec.get("soft_preferences") or []
            raw_hard = spec.get("constraints") or []
            if not isinstance(raw_soft, list) or not all(isinstance(c, dict) for c in raw_soft):
                raise ValueError("soft_preferences must be a list of constraint objects")
            if not isinstance(raw_hard, list) or not all(isinstance(c, dict) for c in raw_hard):
                raise ValueError("constraints must be a list of constraint objects")

            # A wholly empty schema placeholder carries no user semantics. Drop
            # only that exact shape; partially populated constraints still fail
            # validation and must be repaired by the parser.
            def is_empty_placeholder(constraint: Dict[str, Any]) -> bool:
                return (
                    not str(constraint.get("field") or "").strip()
                    and not str(constraint.get("op") or "").strip()
                    and constraint.get("value") is None
                    and not str(constraint.get("source_text") or "").strip()
                )

            raw_soft = [item for item in raw_soft if not is_empty_placeholder(item)]
            raw_hard = [item for item in raw_hard if not is_empty_placeholder(item)]

            def normalize_constraint(constraint: Dict[str, Any]) -> Dict[str, Any]:
                field_name = str(constraint.get("field") or "")
                operator = str(constraint.get("op") or "")
                if field_name not in allowed_fields[entity]:
                    raise ValueError(f"unsupported {entity} constraint field: {field_name!r}")
                # Canonical presence/equality operators are unambiguous for
                # categorical fields. This repairs schema omission only; numeric
                # comparisons remain strict because their direction is semantic.
                if not operator and constraint.get("value") is not None:
                    if field_name in {"services", "tags", "cuisine"}:
                        operator = "contains"
                    elif field_name in {
                        "seat_class", "train_type", "manufacturer", "direct",
                        "brand", "name", "attraction_type", "nearby_location", "star",
                    }:
                        operator = "=="
                    if operator:
                        constraint["op"] = operator
                if operator not in allowed_operators:
                    raise ValueError(f"unsupported constraint operator: {operator!r}")
                value = constraint.get("value")
                if field_name in {"departure_time", "arrival_time", "opening_time", "closing_time"}:
                    minutes = _parse_hhmm(value)
                    if minutes is None:
                        raise ValueError(f"invalid clock constraint: {constraint}")
                    constraint["value"] = minutes
                    constraint["value_type"] = "minutes_since_midnight"
                elif field_name == "duration":
                    minutes = _parse_duration_minutes(value)
                    if minutes is None:
                        raise ValueError(f"invalid duration constraint: {constraint}")
                    constraint["value"] = minutes
                    constraint["value_type"] = "minutes"
                elif field_name in {"price", "rating", "star", "decoration_year", "distance",
                                    "route_distance", "travel_distance"}:
                    if isinstance(value, str) and value.casefold() in {"max", "min", "highest", "lowest"}:
                        raise ValueError(
                            f"numeric field {field_name!r} cannot compare with symbolic value {value!r}; "
                            "encode it as an objective"
                        )
                    constraint["value"] = float(value)
                elif field_name == "attraction_type" and isinstance(value, str):
                    constraint["value"] = re.sub(r"类$", "", value.strip())
                return constraint

            spec["constraints"] = [normalize_constraint(item) for item in raw_hard]
            spec["soft_preferences"] = [normalize_constraint(item) for item in raw_soft]
            objective = spec.get("objective")
            if isinstance(objective, dict):
                objective_field = str(objective.get("field") or "")
                if objective_field not in allowed_fields[entity]:
                    raise ValueError(f"unsupported {entity} objective field: {objective_field!r}")
                if str(objective.get("sense") or "") not in {"min", "max"}:
                    raise ValueError(f"unsupported objective sense: {objective.get('sense')!r}")

        for direction in ("outbound", "return"):
            spec = ir.setdefault("transport", {}).setdefault(direction, {})
            modes = spec.get("allowed_modes") or ["flight", "train"]
            spec["allowed_modes"] = [m for m in modes if m in {"flight", "train"}] or ["flight", "train"]
            spec.setdefault("constraints", [])
            spec.setdefault("objective", None)
            normalize_constraints(spec, "transport")
            for constraint in spec["constraints"]:
                # 动车/高铁 describe the train category, never the cabin seat.
                if constraint.get("field") == "seat_class" and constraint.get("value") in {"动车", "高铁", "普通列车", "普快"}:
                    constraint["field"] = "train_type"
        for section in ("hotel", "attractions"):
            spec = ir.setdefault(section, {})
            spec.setdefault("constraints", [])
            spec.setdefault("objective", None)
            normalize_constraints(spec, "hotel" if section == "hotel" else "attraction")
        ir["attractions"].setdefault("count", None)
        ir["attractions"].setdefault("selection", "default")
        ir["attractions"].setdefault("required_names", [])
        if ir["attractions"]["selection"] == "named_and_top_k":
            # Common neural shorthand outside the public schema: preserve the
            # named set and normalize the additional ranked selection below.
            ir["attractions"]["selection"] = "top_k"
            if not ir["attractions"].get("count"):
                objective = ir["attractions"].get("objective") or {}
                ir["attractions"]["count"] = int(objective.get("top_k") or 1)
        if ir["attractions"]["selection"] not in {"default", "top_k", "all_matching", "named"}:
            raise ValueError(f"unsupported attraction selection: {ir['attractions']['selection']!r}")
        ir["attractions"]["constraints"] = [
            constraint for constraint in ir["attractions"]["constraints"]
            if not (constraint.get("field") == "attraction_type"
                    and constraint.get("value") in {"景点", "旅游景点", "景区"})
        ]
        if ir["attractions"]["selection"] == "all_matching":
            constraints = ir["attractions"]["constraints"]
            sources = {str(c.get("source_text") or "") for c in constraints}
            # Separate "all free attractions" and "all nature attractions"
            # are a union of requested groups, not an accidental AND filter.
            if len(constraints) > 1 and len(sources) > 1:
                ir["attractions"]["selection_groups"] = [[c] for c in constraints]
                ir["attractions"]["constraints"] = []
        if ir["attractions"]["selection"] == "top_k" and not ir["attractions"]["count"]:
            ir["attractions"]["count"] = 1

        # Named must-visits and a separate filtered objective are two selection
        # groups, not one global filter. This is structural IR normalization:
        # it uses only typed fields and never reparses the natural-language query.
        if (ir["attractions"].get("required_names")
                and ir["attractions"].get("selection") in {"named", "top_k"}
                and ir["attractions"].get("constraints")
                and isinstance(ir["attractions"].get("objective"), dict)):
            ir["attractions"].setdefault("required_groups", []).append({
                "count": int(ir["attractions"].get("count") or 1),
                "constraints": copy.deepcopy(ir["attractions"]["constraints"]),
                "objective": copy.deepcopy(ir["attractions"].get("objective")),
            })
            ir["attractions"]["selection"] = "named"
            ir["attractions"]["count"] = None
            ir["attractions"]["constraints"] = []
            ir["attractions"]["objective"] = None

        restaurants = ir.setdefault("restaurants", {})
        if not isinstance(restaurants, dict):
            raise ValueError("restaurants must be an object")
        legacy_constraints = restaurants.pop("constraints", None)
        legacy_objective = restaurants.pop("objective", None)
        restaurants.setdefault("requirements", [])
        if not isinstance(restaurants["requirements"], list) or not all(
                isinstance(requirement, dict) for requirement in restaurants["requirements"]):
            raise ValueError("restaurants.requirements must be a list of objects")
        if not restaurants["requirements"]:
            restaurants["requirements"] = [{
                "required_name": None, "nearby_location": None,
                "constraints": legacy_constraints or [], "objective": legacy_objective,
            }]
        for requirement in restaurants["requirements"]:
            requirement.setdefault("required_name", None)
            requirement.setdefault("nearby_location", None)
            requirement.setdefault("constraints", [])
            requirement.setdefault("objective", None)
            normalize_constraints(requirement, "restaurant")

        ir.setdefault("global_constraints", [])
        ir.setdefault("explicit_preferences", [])

    @staticmethod
    def _validate_constraint_modalities(ir: Dict[str, Any], user_query: str) -> None:
        """Reject hard predicates whose grounded source is explicitly preferential."""
        query = _canonical_text(user_query)

        def soft_context(source_text: Any) -> bool:
            source = _canonical_text(source_text)
            if not source or not query:
                return False
            position = query.find(source)
            context = source if position < 0 else query[max(0, position - 12):position + len(source)]
            return any(marker in context for marker in (
                "非必须", "不强求", "可有可无", "如果有更好", "optional", "nice to have",
            ))

        sections: List[Tuple[str, Dict[str, Any]]] = []
        for direction in ("outbound", "return"):
            sections.append((f"transport.{direction}", ir["transport"][direction]))
        sections.extend((("hotel", ir["hotel"]), ("attractions", ir["attractions"])))
        sections.extend(
            (f"restaurants.requirements.{index}", requirement)
            for index, requirement in enumerate(ir["restaurants"].get("requirements") or [])
        )
        ambiguous = [
            f"{section}.constraints[{index}]"
            for section, spec in sections
            for index, constraint in enumerate(spec.get("constraints") or [])
            if soft_context(constraint.get("source_text"))
        ]
        if ambiguous:
            raise ValueError(
                "explicit soft-modality predicates must be placed in soft_preferences: "
                + ", ".join(ambiguous)
            )

    @staticmethod
    def _semantic_source_phrases(value: Any) -> set[str]:
        """Collect auditable source phrases from every semantic IR node."""
        phrases: set[str] = set()
        if isinstance(value, dict):
            source = value.get("source_text")
            if isinstance(source, str) and source.strip():
                phrases.add(_canonical_text(source))
            for child in value.values():
                phrases.update(SolverInformedTravelAgent._semantic_source_phrases(child))
        elif isinstance(value, list):
            for child in value:
                phrases.update(SolverInformedTravelAgent._semantic_source_phrases(child))
        return phrases

    @staticmethod
    def _phrase_is_grounded(source: str, user_query: str) -> bool:
        source_norm = _canonical_text(source)
        query_norm = _canonical_text(user_query)
        return bool(source_norm) and source_norm in query_norm

    @classmethod
    def _validate_semantic_revision(
        cls, original: Dict[str, Any], candidate: Any, user_query: str,
    ) -> Dict[str, Any]:
        """Validate an untrusted neural IR revision before it affects planning.

        The model may revise semantic sections, but cannot rewrite trip identity,
        dates, party size, budget, or transport modes. New semantic clauses must
        cite the request, and explicit mandatory phrases may not disappear.
        """
        if not isinstance(candidate, dict):
            raise ValueError("corrected_ir must be an object")
        revised = copy.deepcopy(candidate)
        cls._validate_problem_ir(revised)

        protected_trip = (
            "origin", "destination", "start_date", "end_date", "travelers",
            "rooms", "budget_max", "currency",
        )
        for key in protected_trip:
            if revised.get("trip", {}).get(key) != original.get("trip", {}).get(key):
                raise ValueError(f"semantic revision changed protected trip.{key}")
        for direction in ("outbound", "return"):
            old_modes = original.get("transport", {}).get(direction, {}).get("allowed_modes")
            new_modes = revised.get("transport", {}).get(direction, {}).get("allowed_modes")
            if new_modes != old_modes:
                raise ValueError(f"semantic revision changed protected {direction} modes")

        old_sources = cls._semantic_source_phrases(original)
        new_sources = cls._semantic_source_phrases(revised)
        for source in new_sources - old_sources:
            if not cls._phrase_is_grounded(source, user_query):
                raise ValueError(f"new semantic source is not grounded in request: {source!r}")

        mandatory_markers = ("必须", "一定", "务必", "要求", "must", "required", "have to")
        for source in old_sources:
            if any(marker in source for marker in mandatory_markers) and source not in new_sources:
                raise ValueError(f"semantic revision removed mandatory source: {source!r}")

        old_required = set(original.get("attractions", {}).get("required_names") or [])
        new_required = set(revised.get("attractions", {}).get("required_names") or [])
        for name in new_required - old_required:
            if _canonical_text(name) not in _canonical_text(user_query):
                raise ValueError(f"new required attraction is not grounded: {name!r}")
        old_requirements = original.get("restaurants", {}).get("requirements") or []
        old_entities = {
            _canonical_text(requirement.get(field))
            for requirement in old_requirements for field in ("required_name", "nearby_location")
            if requirement.get(field)
        }
        for requirement in revised.get("restaurants", {}).get("requirements") or []:
            for field in ("required_name", "nearby_location"):
                value = requirement.get(field)
                if value and _canonical_text(value) not in old_entities \
                        and _canonical_text(value) not in _canonical_text(user_query):
                    raise ValueError(f"new restaurant {field} is not grounded: {value!r}")

        if json.dumps(revised, ensure_ascii=False, sort_keys=True, default=str) == json.dumps(
                original, ensure_ascii=False, sort_keys=True, default=str):
            raise ValueError("semantic revision is a no-op")
        return revised

    def _semantic_ir_call(
        self, prompt: str, user_query: str, context: Dict[str, Any], metric_name: str,
    ) -> Dict[str, Any]:
        if self.metrics["llm_calls"] >= self.max_llm_calls:
            return {"verdict": "SKIP", "issues": [], "reason": "LLM budget exhausted"}
        response = self._call_llm([
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({
                "original_request": user_query,
                **context,
            }, ensure_ascii=False, default=str)},
        ])
        self.metrics[metric_name] += 1
        return _json_object(response.choices[0].message.content or "")

    def _audit_problem_ir(self, user_query: str, problem_ir: Dict[str, Any]) -> Dict[str, Any]:
        """Audit semantics and allow one checker-guided format/typing repair."""
        if not self.enable_semantic_audit:
            return problem_ir
        context: Dict[str, Any] = {"parsed_ir": problem_ir}
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                result = self._semantic_ir_call(
                    SEMANTIC_AUDIT_PROMPT, user_query, context,
                    "semantic_audit_calls",
                )
                verdict = str(result.get("verdict") or "").upper()
                event = {"stage": "semantic_audit", "verdict": verdict,
                         "issues": result.get("issues") or [], "attempt": attempt + 1}
                if verdict != "REPAIR":
                    self.semantic_events.append(event)
                    return problem_ir
                revised = self._validate_semantic_revision(
                    problem_ir, result.get("corrected_ir"), user_query,
                )
                self.metrics["semantic_audit_repairs"] += 1
                event["applied"] = True
                self.semantic_events.append(event)
                return revised
            except Exception as error:
                last_error = error
                if attempt == 0 and self.metrics["llm_calls"] < self.max_llm_calls:
                    self.metrics["semantic_audit_retries"] += 1
                    context = {
                        "parsed_ir": problem_ir,
                        "previous_audit": result if "result" in locals() else None,
                        "validation_error": str(error),
                        "instruction": "Return a corrected audit that obeys the exact IR schema and field whitelist.",
                    }
                    continue
                break
        self.metrics["semantic_audit_rejected"] += 1
        self.semantic_events.append({
            "stage": "semantic_audit", "verdict": "REJECTED", "error": str(last_error),
        })
        return problem_ir

    def _compile_recall_expansions(
        self, expansions: Any, allowed_constraint_ids: set[str],
    ) -> List[AcquisitionRequest]:
        """Compile constrained neural recall intents into the existing tool queue."""
        assert self.state is not None
        if not isinstance(expansions, list):
            return []
        destination = self.problem_ir["trip"].get("destination")
        official_types = {
            "历史文化", "自然风光", "艺术展览", "城市地标", "主题乐园", "休闲体验",
        }
        grounded_anchors = {
            _canonical_text(record.canonical.get("name")): record
            for kind in ("hotel", "attraction")
            for record in self.state.records_of(kind)
            if record.canonical.get("name") and _coord(
                record.canonical.get("lat"), record.canonical.get("lon"))
        }
        requests: Dict[str, AcquisitionRequest] = {}
        for expansion in expansions:
            if not isinstance(expansion, dict):
                continue
            constraint_id = str(expansion.get("constraint_id") or "")
            if constraint_id not in allowed_constraint_ids:
                continue
            kind = str(expansion.get("kind") or "")
            values = expansion.get("values") or []
            if not isinstance(values, list):
                continue
            accepted_for_constraint = False
            if kind == "additional_anchor" and constraint_id.startswith("restaurants.requirement."):
                for value in values:
                    record = grounded_anchors.get(_canonical_text(value))
                    if record is None:
                        continue
                    coordinate = _coord(record.canonical.get("lat"), record.canonical.get("lon"))
                    if not coordinate:
                        continue
                    lat, lon = coordinate.split(",", 1)
                    request = AcquisitionRequest(
                        "recommend_restaurants", f"reflection recall for {constraint_id}",
                        {"latitude": lat, "longitude": lon,
                         "nearby_location": record.canonical.get("name")},
                        f"restaurants:recovery:{constraint_id}:{coordinate}", 85,
                    )
                    requests[request.key] = request
                    accepted_for_constraint = True
            elif kind == "attraction_type" and constraint_id == "attractions.coverage":
                for value in values:
                    attraction_type = re.sub(r"类$", "", str(value).strip())
                    if attraction_type not in official_types:
                        continue
                    request = AcquisitionRequest(
                        "recommend_attractions", f"reflection recall for {constraint_id}",
                        {"city": destination, "attraction_type": attraction_type},
                        f"attractions:{destination}:{attraction_type}", 85,
                    )
                    requests[request.key] = request
                    accepted_for_constraint = True
            elif kind == "unrestricted_attractions" and constraint_id == "attractions.coverage":
                request = AcquisitionRequest(
                    "recommend_attractions", f"reflection recall for {constraint_id}",
                    {"city": destination, "attraction_type": None},
                    f"attractions:{destination}:all", 85,
                )
                requests[request.key] = request
                accepted_for_constraint = True
        return [request for request in requests.values()
                if self.tools.has(request.tool) and self.state.can_attempt(request.key)]

    def _reflect_on_search(
        self, user_query: str, result: SolveResult,
        best_incumbent: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[AcquisitionRequest]]:
        """Bounded solver-conflict reflection that can only propose legal retrieval intents."""
        if (not self.enable_reflection
                or self.metrics["reflection_calls"] >= self.max_reflections
                or result.status == PROVEN_INFEASIBLE):
            return "KEEP", []
        constraint_by_id = {
            constraint.constraint_id: constraint
            for constraint in (self.constraint_engine.constraints if self.constraint_engine else [])
        }
        recall_sensitive_ids = {
            constraint.constraint_id for constraint in constraint_by_id.values()
            if constraint.kind in {"attraction_coverage", "restaurant_requirement"}
        }
        conflict = result.conflict or {"kind": result.status, "reason": result.reason}
        conflict_kind = str(conflict.get("kind") or "")
        if conflict_kind in {"no_attraction_candidates", "template_incomplete"}:
            allowed_ids = {cid for cid in recall_sensitive_ids if cid == "attractions.coverage"}
        elif conflict_kind == "no_restaurant_candidates":
            allowed_ids = {cid for cid in recall_sensitive_ids if cid.startswith("restaurants.")}
        elif conflict_kind in {"coverage_infeasible", "temporal_infeasible", "verification_failed",
                               "uncertified_symbolic_conflict"}:
            allowed_ids = recall_sensitive_ids
        else:
            allowed_ids = set()
        anchors = sorted({
            str(record.canonical.get("name"))
            for kind in ("hotel", "attraction")
            for record in (self.state.records_of(kind) if self.state else [])
            if record.canonical.get("name") and _coord(
                record.canonical.get("lat"), record.canonical.get("lon"))
        })[:30]
        try:
            response = self._semantic_ir_call(
                REFLECTION_PROMPT, user_query,
                {
                    "current_ir": self.problem_ir,
                    "failure_status": result.status,
                    "solver_conflict": conflict,
                    "state_summary": self.state.compact_summary() if self.state else None,
                    "search_summary": {
                        "solver_rounds": self.metrics["solver_rounds"],
                        "master_assignments": self.metrics["master_assignments"],
                        "frontier_states": self.metrics["frontier_states"],
                        "tool_calls": self.metrics["tool_calls"],
                    },
                    "best_incumbent": best_incumbent,
                    "allowed_recall_constraint_ids": sorted(allowed_ids),
                    "allowed_grounded_anchors": anchors,
                    "allowed_attraction_types": [
                        "历史文化", "自然风光", "艺术展览", "城市地标", "主题乐园", "休闲体验",
                    ],
                },
                "reflection_calls",
            )
            diagnosis = str(response.get("diagnosis") or "SEARCH_EXHAUSTED").upper()
            action = str(response.get("action") or "KEEP").upper()
            if action == "KEEP":
                self.semantic_events.append({
                    "stage": "reflection", "diagnosis": diagnosis,
                    "action": "KEEP", "reason": response.get("reason"),
                    "conflict": conflict,
                })
                return "KEEP", []
            if action == "RECHECK_SEMANTICS":
                revised = self._validate_semantic_revision(
                    self.problem_ir, response.get("corrected_ir"), user_query,
                )
                signature = _stable_id("reflection", action, revised)
                if signature in self._reflection_signatures:
                    self.metrics["reflection_duplicate_actions"] += 1
                    return "KEEP", []
                self._reflection_signatures.add(signature)
                self.problem_ir = revised
                if self.state is not None:
                    self.state.problem_ir = revised
                self.constraint_engine = IncrementalConstraintEngine(self)
                self.metrics["reflection_actions_accepted"] += 1
                self.semantic_events.append({
                    "stage": "reflection", "diagnosis": diagnosis,
                    "action": action, "reason": response.get("reason"), "applied": True,
                })
                return "REPAIRED", []
            if action == "EXPAND_RECALL":
                requests = self._compile_recall_expansions(
                    response.get("recall_expansions") or [], allowed_ids,
                )
                signature = _stable_id("reflection", action, sorted(req.key for req in requests))
                if not requests or signature in self._reflection_signatures:
                    if signature in self._reflection_signatures:
                        self.metrics["reflection_duplicate_actions"] += 1
                    return "KEEP", []
                self._reflection_signatures.add(signature)
                self.metrics["reflection_actions_accepted"] += 1
                self.semantic_events.append({
                    "stage": "reflection", "diagnosis": diagnosis,
                    "action": action, "reason": response.get("reason"),
                    "accepted_request_keys": [request.key for request in requests],
                    "conflict": conflict,
                })
                return "EXPANDED", requests
            raise ValueError(f"unsupported reflection action: {action!r}")
        except Exception as error:
            self.metrics["recovery_rejected"] += 1
            self.semantic_events.append({
                "stage": "reflection", "action": "REJECTED",
                "error": str(error), "conflict": conflict,
            })
            return "KEEP", []

    # ---------------------------------------------------------------------
    # Tool argument compilation and execution
    # ---------------------------------------------------------------------

    def _compile_tool_args(self, request: AcquisitionRequest) -> Dict[str, Any]:
        return OfficialToolAdapters.build(request, self.state)

    @staticmethod
    def _tool_query_is_exhaustive(request: AcquisitionRequest) -> bool:
        """Public protocol contract used only for certificate eligibility."""
        return request.tool in {
            "query_train_info", "query_flight_info", "query_hotel_info",
            "query_attraction_details", "query_restaurant_details",
            "query_road_route_info", "search_location",
        }

    def _execute_request(self, request: AcquisitionRequest) -> bool:
        assert self.state is not None
        if self.state.is_resolved(request.key):
            self.metrics["duplicate_tool_calls_avoided"] += 1
            return False
        if not self.state.can_attempt(request.key):
            self.metrics["tool_retry_exhausted"] += 1
            return False
        if self.state.request_attempts.get(request.key, 0) > 0:
            self.metrics["tool_retries"] += 1
        if self.metrics["tool_calls"] >= self.max_tool_calls:
            raise RuntimeError("tool-call budget exhausted")
        if not self.tools.has(request.tool):
            payload = {"error": "tool unavailable"}
            self.state.add_evidence(request, {}, payload)
            self.state.record_query_outcome(request, payload, [], False)
            self.metrics["tool_errors"] += 1
            return False

        try:
            args = self._compile_tool_args(request)
        except ValueError as e:
            payload = {"error": str(e)}
            self.state.add_evidence(request, {}, payload)
            self.state.record_query_outcome(request, payload, [], False)
            self.metrics["tool_errors"] += 1
            return False
        payload = self.tools.call(request.tool, args)
        self.metrics["tool_calls"] += 1
        self.metrics["tool_calls_by_name"][request.tool] = self.metrics["tool_calls_by_name"].get(request.tool, 0) + 1
        if request.tool == "query_road_route_info":
            self.metrics["route_queries"] += 1
        if isinstance(payload, dict) and payload.get("error"):
            self.metrics["tool_errors"] += 1

        self.state.add_evidence(request, args, payload)
        # Never canonicalize a protocol/tool error into an entity or route record.
        if _payload_has_error(payload):
            self.state.record_query_outcome(request, payload, [], False)
            return False

        records = Canonicalizer.canonicalize(request.tool, payload, self.language)
        self.state.record_query_outcome(
            request, payload, records, self._tool_query_is_exhaustive(request),
        )
        outcome = self.state.query_outcomes[request.key]
        if outcome.status == QUERY_PARSE_ERROR:
            self.metrics["tool_parse_errors"] += 1
            return False

        # Route payloads often omit endpoints; recover them from the controller request.
        if request.tool == "query_road_route_info":
            for rec in records:
                # Index routes by semantic entity names while retaining the exact
                # coordinate arguments in raw evidence/provenance.
                rec.canonical["origin"] = request.args_hint.get("from_location") or rec.canonical.get("origin")
                rec.canonical["destination"] = request.args_hint.get("to_location") or rec.canonical.get("destination")
                if not rec.canonical.get("name"):
                    rec.canonical["name"] = f"{rec.canonical.get('origin')} -> {rec.canonical.get('destination')}"
        for rec in records:
            self.state.add_record(rec)
        return True

    # ---------------------------------------------------------------------
    # Information-acquisition policy
    # ---------------------------------------------------------------------

    def _initial_requests(self) -> List[AcquisitionRequest]:
        trip = self.problem_ir["trip"]
        origin = trip.get("origin")
        dest = trip.get("destination")
        start_date = trip.get("start_date")
        end_date = trip.get("end_date")
        common = {
            "origin": origin,
            "destination": dest,
            "city": dest,
            "start_date": start_date,
            "end_date": end_date,
            "rooms": trip.get("rooms", 1),
            "travelers": trip.get("travelers", 1),
        }
        reqs: List[AcquisitionRequest] = []
        for direction, date, frm, to in (
            ("outbound", start_date, origin, dest),
            ("return", end_date, dest, origin),
        ):
            modes = self.problem_ir["transport"][direction]["allowed_modes"]
            constraints = self.problem_ir["transport"][direction].get("constraints") or []
            seat_class = next((c.get("value") for c in constraints if c.get("field") == "seat_class" and c.get("op") == "=="), None)
            for mode in modes:
                tool = "query_flight_info" if mode == "flight" else "query_train_info"
                reqs.append(AcquisitionRequest(
                    tool=tool,
                    purpose=f"retrieve {direction} {mode} candidates",
                    args_hint={**common, "origin": frm, "destination": to, "date": date, "seat_class": seat_class},
                    key=f"{direction}:{mode}:{frm}:{to}:{date}",
                    priority=100,
                ))
        hotel_constraints = (self.problem_ir.get("hotel") or {}).get("constraints") or []
        hotel_star = next((c.get("value") for c in hotel_constraints if c.get("field") == "star" and c.get("op") == "=="), None)
        # The tool's brand filter uses literal database labels (e.g. "全季"),
        # while the user may say "全季酒店". Retrieve the full public tool result
        # and apply the normalized constraint in the controller.
        hotel_brand = None
        attraction_constraints = (self.problem_ir.get("attractions") or {}).get("constraints") or []
        attraction_type = next((c.get("value") for c in attraction_constraints if c.get("field") == "attraction_type" and c.get("op") == "=="), None)
        reqs.extend([
            AcquisitionRequest("query_hotel_info", "retrieve hotel candidates",
                {**common, "city": dest, "hotel_star": hotel_star, "hotel_brand": hotel_brand},
                f"hotels:{dest}:{start_date}:{end_date}:{hotel_star}:{hotel_brand}", 90),
            AcquisitionRequest("recommend_attractions", "retrieve attraction candidates",
                {**common, "city": dest, "attraction_type": attraction_type},
                f"attractions:{dest}:{attraction_type or 'all'}", 80),
        ])
        if attraction_type and (
                self.problem_ir["attractions"].get("selection") == "top_k"
                or self.problem_ir["attractions"].get("required_groups")):
            reqs.append(AcquisitionRequest(
                "recommend_attractions", "retrieve unrestricted candidates for ordinary itinerary slots",
                {**common, "city": dest, "attraction_type": None},
                f"attractions:{dest}:all", 79,
            ))
        return sorted(reqs, key=lambda r: -r.priority)

    def _restaurant_requirements(self) -> List[Dict[str, Any]]:
        return list((self.problem_ir.get("restaurants") or {}).get("requirements") or [])

    def _location_request(self, name: str, purpose: str) -> AcquisitionRequest:
        return AcquisitionRequest(
            "search_location", purpose, {"name": name, "place_name": name},
            f"location:{_canonical_text(name)}", 100,
        )

    def _restaurant_requests(self) -> List[AcquisitionRequest]:
        """Acquire each dining requirement around its grounded anchor."""
        assert self.state is not None
        reqs: List[AcquisitionRequest] = []
        requirements = self._restaurant_requirements()
        for requirement in requirements:
            required_name = requirement.get("required_name")
            if required_name:
                reqs.append(AcquisitionRequest(
                    "query_restaurant_details", "retrieve explicitly requested restaurant",
                    {"name": required_name}, f"details:restaurant:{_canonical_text(required_name)}", 110,
                ))
                continue
            nearby = requirement.get("nearby_location")
            if nearby:
                coordinate = self.state.coordinates(str(nearby))
                if coordinate is None:
                    reqs.append(self._location_request(str(nearby), "ground restaurant-nearby anchor"))
                    continue
                lat, lon = coordinate.split(",", 1)
                reqs.append(AcquisitionRequest(
                    "recommend_restaurants", f"retrieve restaurants near {nearby}",
                    {"latitude": lat, "longitude": lon, "nearby_location": nearby},
                    f"restaurants:{_canonical_text(nearby)}:{coordinate}", 100,
                ))
                continue

        # Meals beyond explicit dining requirements need a varied, grounded
        # pool. Query a few already discovered attraction coordinates; the
        # official recommendation tool determines the restaurants.
        seen_coords = {
            f"{r.args_hint.get('latitude')},{r.args_hint.get('longitude')}"
            for r in reqs if r.tool == "recommend_restaurants"
        }
        for attraction in self._rank_records("attraction")[:3]:
            coordinate = _coord(attraction.canonical.get("lat"), attraction.canonical.get("lon"))
            if not coordinate or coordinate in seen_coords:
                continue
            seen_coords.add(coordinate)
            lat, lon = coordinate.split(",", 1)
            reqs.append(AcquisitionRequest(
                "recommend_restaurants", "acquire grounded options for ordinary meals",
                {"latitude": lat, "longitude": lon, "nearby_location": attraction.canonical.get("name")},
                f"restaurants:generic:{coordinate}", 50,
            ))
        return [r for r in reqs if self.tools.has(r.tool)]

    def _detail_requests(self) -> List[AcquisitionRequest]:
        """Ask detail tools for candidates whose opening/duration facts are still unknown."""
        assert self.state is not None
        reqs: List[AcquisitionRequest] = []
        required_names = list((self.problem_ir.get("attractions") or {}).get("required_names") or [])
        required_attractions = {_canonical_text(x) for x in required_names}
        known_attractions = {
            _canonical_text(record.canonical.get("name"))
            for record in self.state.records_of("attraction")
        }
        if self.tools.has("query_attraction_details"):
            for name in required_names:
                if _canonical_text(name) not in known_attractions:
                    reqs.append(AcquisitionRequest(
                        "query_attraction_details", "retrieve explicitly required attraction",
                        {"name": name, "query": name,
                         "city": self.problem_ir["trip"].get("destination")},
                        f"details:attraction:{_canonical_text(name)}", 110,
                    ))
        for kind, tool in (("attraction", "query_attraction_details"), ("restaurant", "query_restaurant_details")):
            if not self.tools.has(tool):
                continue
            ranked = self._rank_records(kind)
            selection = (self.problem_ir.get("attractions") or {}).get("selection")
            if kind == "attraction" and selection == "top_k":
                ranked = sorted(
                    self.state.records_of("attraction"),
                    key=lambda r: (r.canonical.get("rating") is None,
                                   -(r.canonical.get("rating") or 0.0), r.record_id),
                )
            selected = ranked if kind == "attraction" and selection in {"top_k", "all_matching"} else ranked[:self._pool_limit(kind)]
            if kind == "attraction":
                selected_ids = {r.record_id for r in selected}
                selected += [r for r in ranked if _canonical_text(r.canonical.get("name")) in required_attractions and r.record_id not in selected_ids]
            for rec in selected:
                c = rec.canonical
                if c.get("opening_time") is not None and (kind == "restaurant" or c.get("min_duration") is not None):
                    continue
                name = c.get("name")
                if not name:
                    continue
                reqs.append(AcquisitionRequest(
                    tool, f"retrieve details for {kind} {name}",
                    {"name": name, "query": name, "city": self.problem_ir["trip"].get("destination")},
                    f"details:{kind}:{_canonical_text(name)}", 60,
                ))
        return reqs

    def _pool_limit(self, kind: str) -> int:
        defaults = {"hotel": 5, "attraction": 8, "restaurant": 6, "flight": 8, "train": 8}
        env = f"TRAVEL_{kind.upper()}_POOL"
        try:
            if env in os.environ:
                return max(1, int(os.environ[env]))
            if kind == "attraction":
                return max(defaults[kind], 3 * self._trip_days())
            if kind == "restaurant":
                return max(defaults[kind], 4 * self._trip_days())
            return defaults.get(kind, 6)
        except ValueError:
            return defaults.get(kind, 6)

    def _record_field(self, rec: EvidenceRecord, field: str) -> Any:
        aliases = {
            "cost": "price", "fare": "price", "ticket_price": "price",
            "score": "rating", "stars": "star", "hotel_star": "star",
            "service": "services", "hotel_service": "services",
            "renovation_year": "decoration_year", "decoration_time": "decoration_year",
            "category": "attraction_type", "type": "attraction_type",
            "tag": "tags", "restaurant_tag": "tags",
            "travel_time": "duration", "time": "duration",
        }
        field = aliases.get(field, field)
        if rec.kind == "restaurant":
            if field in {"services", "service"}:
                field = "tags"
            if field == "queue_online":
                return any("线上取号" in _canonical_text(x) for x in rec.canonical.get("tags") or [])
        value = rec.canonical.get(field)
        if value is not None:
            return value
        return _deep_get(rec.raw, field)

    def _status_against(self, rec: EvidenceRecord, constraints: Sequence[Dict[str, Any]]) -> str:
        unknown = False
        for constraint in constraints:
            field_name = str(constraint.get("field", ""))
            actual = self._record_field(rec, field_name)
            expected = constraint.get("value")
            op = str(constraint.get("op", "=="))
            if field_name == "brand" and isinstance(expected, str):
                expected = re.sub(r"(?:酒店|hotel)$", "", expected, flags=re.I).strip()
                op = "contains" if op == "==" else op
            if rec.kind == "hotel" and field_name in {"services", "service"}:
                actual = " ".join(str(x) for x in rec.canonical.get("services") or [])
                for old, new in (("电视可投屏", "投屏"), ("电视投屏", "投屏"),
                                 ("免费停车场", "免费停车"), ("有停车场", "停车")):
                    actual = actual.replace(old, new)
                    if isinstance(expected, str):
                        expected = expected.replace(old, new)
                op = "contains" if op == "==" else op
            if rec.kind == "restaurant":
                if field_name in {"tags", "services", "service"}:
                    tokens = rec.canonical.get("tags") or []
                    actual = " ".join(str(x) for x in tokens) + " " + str(rec.canonical.get("cuisine") or "")
                    if isinstance(expected, str):
                        expected = expected.replace("服务", "").replace("有", "")
                        expected = re.sub(r"(?:餐厅|菜馆|饭店)$", "", expected).strip()
                    op = "contains" if op == "==" else op
                elif field_name == "cuisine":
                    # Natural requests commonly use an establishment suffix
                    # (e.g. 清真菜馆) while the canonical cuisine is 清真菜, or vice
                    # versa.  Compare their semantic category rather than the suffix.
                    if isinstance(expected, str):
                        expected = re.sub(r"(?:餐厅|菜馆|饭店)$", "", expected).strip()
                    if isinstance(actual, str):
                        actual = re.sub(r"(?:餐厅|菜馆|饭店)$", "", actual).strip()
                    if op == "==":
                        op = "contains"
            result = _compare(actual, op, expected)
            if result is False:
                return UNSAT
            if result is None:
                unknown = True
        return UNKNOWN if unknown else SAT

    def _rank_records(self, kind: str, direction: Optional[str] = None) -> List[EvidenceRecord]:
        assert self.state is not None
        records = self.state.records_of(kind)
        if kind in {"flight", "train"} and direction:
            spec = self.problem_ir["transport"][direction]
            constraints = spec.get("constraints") or []
            objective = spec.get("objective")
            trip = self.problem_ir["trip"]
            expected_origin = trip.get("origin") if direction == "outbound" else trip.get("destination")
            expected_destination = trip.get("destination") if direction == "outbound" else trip.get("origin")
            expected_date = trip.get("start_date") if direction == "outbound" else trip.get("end_date")
            records = [
                rec for rec in records
                if _canonical_text(rec.canonical.get("origin")) == _canonical_text(expected_origin)
                and _canonical_text(rec.canonical.get("destination")) == _canonical_text(expected_destination)
                and str(rec.canonical.get("departure_date")) == str(expected_date)
                and (direction != "outbound" or str(rec.canonical.get("arrival_date")) == str(expected_date))
                and (direction != "return" or str(rec.canonical.get("arrival_date")) == str(expected_date))
            ]
        elif kind == "restaurant":
            spec = {}
            constraints = []
            objective = None
        else:
            section = "attractions" if kind == "attraction" else "restaurants" if kind == "restaurant" else kind
            spec = self.problem_ir.get(section) or {}
            constraints = spec.get("constraints") or []
            objective = spec.get("objective")

        if kind in {"flight", "train"}:
            travelers = int(self.problem_ir["trip"].get("travelers", 1))
            records = [r for r in records if (r.canonical.get("seat_status") or 0) >= travelers]
        filtered = [r for r in records if self._status_against(r, constraints) != UNSAT]
        if objective and isinstance(objective, dict):
            field_name = str(objective.get("field"))
            sense = str(objective.get("sense", "min"))
            def obj_value(r: EvidenceRecord) -> float:
                v = _to_float(self._record_field(r, field_name))
                if v is None:
                    return float("inf") if sense == "min" else float("-inf")
                return v
            return sorted(filtered, key=obj_value, reverse=(sense == "max"))

        # Stable defaults are not claimed as user objectives; they only prioritize search.
        if kind in {"flight", "train", "hotel", "restaurant"}:
            return sorted(filtered, key=lambda r: (r.canonical.get("price") is None, r.canonical.get("price") or float("inf"), r.record_id))
        if kind == "attraction":
            return sorted(filtered, key=lambda r: (r.canonical.get("rating") is None, -(r.canonical.get("rating") or 0.0), r.record_id))
        return filtered

    def _restaurants_for_requirement(self, requirement: Dict[str, Any]) -> List[EvidenceRecord]:
        assert self.state is not None
        records = self.state.records_of("restaurant")
        required_name = requirement.get("required_name")
        nearby = requirement.get("nearby_location")
        constraints = requirement.get("constraints") or []
        filtered: List[EvidenceRecord] = []
        for rec in records:
            if required_name and _canonical_text(rec.canonical.get("name")) != _canonical_text(required_name):
                continue
            if nearby and not self._restaurant_matches_anchor(rec, str(nearby)):
                continue
            if self._status_against(rec, constraints) == UNSAT:
                continue
            filtered.append(rec)
        objective = requirement.get("objective")
        if isinstance(objective, dict):
            field_name, sense = str(objective.get("field")), str(objective.get("sense", "min"))
            def objective_value(record: EvidenceRecord) -> Optional[float]:
                if field_name in {"distance", "route_distance", "travel_distance"} and nearby:
                    route = self.state.route(str(nearby), str(record.canonical.get("name")))
                    return _to_float(route.canonical.get("distance")) if route else None
                return _to_float(self._record_field(record, field_name))
            filtered.sort(
                key=lambda r: objective_value(r)
                if objective_value(r) is not None
                else (float("inf") if sense == "min" else float("-inf")),
                reverse=(sense == "max"),
            )
            # A superlative is a hard selection requirement in this benchmark.
            if filtered and not objective.get("soft"):
                best = objective_value(filtered[0])
                filtered = [r for r in filtered if objective_value(r) == best]
        else:
            filtered.sort(key=lambda r: (r.canonical.get("price") is None, r.canonical.get("price") or float("inf")))
        soft = requirement.get("soft_preferences") or []
        if soft and filtered:
            scores = {r.record_id: sum(self._status_against(r, [pref]) == SAT for pref in soft) for r in filtered}
            best_score = max(scores.values())
            if best_score > 0:
                filtered = [r for r in filtered if scores[r.record_id] == best_score]
        return filtered

    @staticmethod
    def _restaurant_matches_anchor(record: EvidenceRecord, anchor: str) -> bool:
        grounded = list(record.canonical.get("nearby_locations") or [])
        grounded.append(record.canonical.get("nearby_location"))
        expected = _canonical_text(anchor)
        return any(_canonical_text(value) == expected for value in grounded if value)

    def _anchor_name(self, transport: EvidenceRecord, outbound: bool) -> Optional[str]:
        key = "arrival_station" if outbound else "departure_station"
        return transport.canonical.get(key) or transport.canonical.get("destination" if outbound else "origin")

    # ---------------------------------------------------------------------
    # Day-template construction
    # ---------------------------------------------------------------------

    def _trip_days(self) -> int:
        trip = self.problem_ir["trip"]
        start = _parse_date(trip.get("start_date"))
        end = _parse_date(trip.get("end_date"))
        if start and end and end >= start:
            return (end - start).days + 1
        return 3

    def _closed_on_day(self, rec: EvidenceRecord, day: int) -> bool:
        closed = str(rec.canonical.get("closing_dates") or "")
        if not closed:
            return False
        start = _parse_date(self.problem_ir["trip"].get("start_date"))
        if start is None:
            return False
        weekday = (start.weekday() + day - 1) % 7
        chinese_day = ("一", "二", "三", "四", "五", "六", "日")[weekday]
        return f"周{chinese_day}" in closed or f"星期{chinese_day}" in closed

    def _required_attraction_count(self, days: int) -> int:
        spec = self.problem_ir["attractions"]
        explicit = spec.get("count")
        if spec.get("selection") == "all_matching" and self.state is not None:
            rows = self._rank_records("attraction")
            groups = spec.get("selection_groups")
            return sum(any(self._status_against(r, group) == SAT for group in groups) for r in rows) if groups else len(rows)
        required_names = spec.get("required_names") or []
        # Benchmark-compatible density prior, not a user preference/objective.
        baseline = max(1, 2 * max(1, days - 1))
        return max(baseline, int(explicit or 0), len(required_names), 3 if days == 2 else 0)

    def _build_templates(
        self,
        outbound: EvidenceRecord,
        return_trip: EvidenceRecord,
        hotel: EvidenceRecord,
        attraction_pool: List[EvidenceRecord],
        restaurant_pool: List[EvidenceRecord],
    ) -> List[DayTemplate]:
        self._template_diagnostic = {}
        days = self._trip_days()
        hotel_name = hotel.canonical.get("name")
        arrival_anchor = self._anchor_name(outbound, outbound=True)
        departure_anchor = self._anchor_name(return_trip, outbound=False)
        if not hotel_name or not arrival_anchor or not departure_anchor:
            return []

        out_arr = outbound.canonical.get("arrival_time")
        ret_dep = return_trip.canonical.get("departure_time")
        if out_arr is None or ret_dep is None:
            return []
        arrival_route_minutes, _, _ = self._route_values(arrival_anchor, hotel_name)
        if arrival_route_minutes is None:
            return []
        hotel_ready_time = int(out_arr) + 45 + int(arrival_route_minutes) + 30

        attraction_ids = [r.record_id for r in attraction_pool]
        required_names = [
            _canonical_text(x) for x in self.problem_ir["attractions"].get("required_names") or []
        ]
        required_attraction_ids = [
            r.record_id for name in required_names for r in attraction_pool
            if _canonical_text(r.canonical.get("name")) == name
        ]
        if self.problem_ir["attractions"].get("selection") == "top_k":
            spec = self.problem_ir["attractions"]
            eligible = [
                r for r in attraction_pool
                if self._status_against(r, spec.get("constraints") or []) == SAT
            ]
            objective = spec.get("objective") or {"field": "rating", "sense": "max"}
            field_name, sense = objective.get("field"), objective.get("sense", "max")
            eligible.sort(
                key=lambda r: _to_float(self._record_field(r, field_name))
                if _to_float(self._record_field(r, field_name)) is not None
                else (float("inf") if sense == "min" else float("-inf")),
                reverse=sense == "max",
            )
            top_k = int(spec.get("count") or 1)
            required_attraction_ids.extend(
                r.record_id for r in eligible[:top_k]
                if r.record_id not in required_attraction_ids
            )
        for group in self.problem_ir["attractions"].get("required_groups") or []:
            eligible = [
                r for r in attraction_pool
                if self._status_against(r, group.get("constraints") or []) == SAT
            ]
            group_count = int(group.get("count") or 1)
            if len(eligible) < group_count:
                self._template_diagnostic = {"reason": "required_attraction_group_missing", "group": group,
                                             "eligible": len(eligible)}
                return []
            objective = group.get("objective") or {"field": "rating", "sense": "max"}
            field_name, sense = objective.get("field"), objective.get("sense", "max")
            eligible.sort(
                key=lambda r: _to_float(self._record_field(r, field_name))
                if _to_float(self._record_field(r, field_name)) is not None
                else (float("inf") if sense == "min" else float("-inf")),
                reverse=sense == "max",
            )
            required_attraction_ids.extend(
                r.record_id for r in eligible[:group_count]
                if r.record_id not in required_attraction_ids
            )
        if self.problem_ir["attractions"].get("selection") == "all_matching":
            required_attraction_ids.extend(rid for rid in attraction_ids if rid not in required_attraction_ids)
        required_attraction_ids.sort(
            key=lambda rid: not self._closed_on_day(self.state.records[rid], days)
        )
        restaurant_ids = [r.record_id for r in restaurant_pool]
        restaurant_requirements = self._restaurant_requirements()
        restaurant_slot_sets = [
            [r.record_id for r in self._restaurants_for_requirement(req)[:self._pool_limit("restaurant")]]
            for req in restaurant_requirements
        ]
        def attraction_slot(slot_id: str, day: int) -> ActivitySlot:
            return ActivitySlot(slot_id, day, "attraction", [
                rid for rid in attraction_ids if not self._closed_on_day(self.state.records[rid], day)
            ])

        def meal_slot(slot_id: str, day: int, kind: str, earliest: int, latest: int) -> ActivitySlot:
            return ActivitySlot(slot_id, day, kind, list(restaurant_ids), earliest, latest)
        target_attractions = self._required_attraction_count(days)
        # Preserve a visit on a late-departure return day while full tourist
        # days retain two visits. The master decides entities across all days.
        return_visit = int(ret_dep > RETURN_DAY_ATTRACTION_THRESHOLD)
        if days > 2 and return_visit:
            target_attractions = max(target_attractions, 2 * (days - 1) + 1)
        target_attractions = min(target_attractions, max(1, 3 * days))

        templates: List[DayTemplate] = []
        remaining_attr_slots = target_attractions
        for day in range(1, days + 1):
            slots: List[ActivitySlot] = []
            if day == 1:
                # Arrival station/airport -> hotel and check-in are fixed prelude
                # activities. The optimization day starts from the hotel only after
                # those grounded steps, preventing teleportation before sightseeing.
                start_anchor = hotel_name
                start_time = hotel_ready_time
                end_anchor = hotel_name
                end_deadline = 23 * 60
                # Arrival rules from benchmark prompt.
                if out_arr < 10 * 60:
                    if hotel_ready_time >= 11 * 60:
                        slots.append(meal_slot(f"d{day}:lunch", day, "lunch", 11 * 60, 16 * 60))
                    if remaining_attr_slots > return_visit:
                        slots.append(attraction_slot(f"d{day}:a1", day))
                        remaining_attr_slots -= 1
                    if hotel_ready_time < 11 * 60:
                        slots.append(meal_slot(f"d{day}:lunch", day, "lunch", 11 * 60, 14 * 60))
                    if remaining_attr_slots > return_visit:
                        slots.append(attraction_slot(f"d{day}:a2", day))
                        remaining_attr_slots -= 1
                    slots.append(meal_slot(f"d{day}:dinner", day, "dinner", 17 * 60, 21 * 60))
                elif out_arr < 15 * 60:
                    if remaining_attr_slots > return_visit:
                        slots.append(attraction_slot(f"d{day}:a1", day))
                        remaining_attr_slots -= 1
                    slots.append(meal_slot(f"d{day}:dinner", day, "dinner", 17 * 60, 21 * 60))
                else:
                    if hotel_ready_time < 20 * 60:
                        slots.append(meal_slot(f"d{day}:dinner", day, "dinner", 17 * 60, 21 * 60))
            elif day == days:
                start_anchor = hotel_name
                # Early departures require an early hotel checkout/transfer.
                start_time = max(0, min(8 * 60 + 30, int(ret_dep) - 180))
                end_anchor = departure_anchor
                end_deadline = int(ret_dep) - 60  # boarding/security buffer starts here
                if ret_dep >= 9 * 60 and remaining_attr_slots > 0:
                    slots.append(attraction_slot(f"d{day}:a1", day))
                    remaining_attr_slots -= 1
                if ret_dep >= 13 * 60:
                    slots.append(meal_slot(f"d{day}:lunch", day, "lunch", 11 * 60, 14 * 60))
                if ret_dep >= RETURN_DAY_ATTRACTION_THRESHOLD and remaining_attr_slots > 0:
                    slots.append(attraction_slot(f"d{day}:a2", day))
                    remaining_attr_slots -= 1
                if ret_dep >= 19 * 60 + 30:
                    slots.append(meal_slot(f"d{day}:dinner", day, "dinner", 17 * 60, 20 * 60))
            else:
                start_anchor = hotel_name
                start_time = 8 * 60 + 30
                end_anchor = hotel_name
                end_deadline = 22 * 60 + 30
                if remaining_attr_slots > return_visit:
                    slots.append(attraction_slot(f"d{day}:a1", day))
                    remaining_attr_slots -= 1
                slots.append(meal_slot(f"d{day}:lunch", day, "lunch", 11 * 60, 14 * 60))
                if remaining_attr_slots > return_visit:
                    slots.append(attraction_slot(f"d{day}:a2", day))
                    remaining_attr_slots -= 1
                slots.append(meal_slot(f"d{day}:dinner", day, "dinner", 17 * 60, 21 * 60))

            templates.append(DayTemplate(day, start_anchor, end_anchor, start_time, end_deadline, slots))
        # Add only the extra visits needed to cover explicit all-matching/top-k
        # requests; distribute them across usable days before global assignment.
        while remaining_attr_slots > 0:
            eligible = [day for day in templates
                        if (day.day != days or ret_dep >= RETURN_DAY_ATTRACTION_THRESHOLD)
                        and (day.day != 1 or out_arr < 12 * 60)
                        and sum(slot.kind == "attraction" for slot in day.slots) < 3]
            if not eligible:
                break
            eligible.sort(key=lambda day: (sum(slot.kind == "attraction" for slot in day.slots), day.day))
            day = eligible[0]
            number = 1 + sum(slot.kind == "attraction" for slot in day.slots)
            extra = attraction_slot(f"d{day.day}:a{number}", day.day)
            dinner_idx = next((i for i, slot in enumerate(day.slots) if slot.kind == "dinner"), len(day.slots))
            day.slots.insert(dinner_idx, extra)
            remaining_attr_slots -= 1
        # Assign named visits to actual attraction slots. When check-in consumes
        # the first morning and a requested meal is near that visit, place the
        # visit on the return day so the meal can follow it.
        attraction_slots = [slot for template in templates for slot in template.slots if slot.kind == "attraction"]
        if len(required_attraction_ids) > len(attraction_slots):
            self._template_diagnostic = {"reason": "required_attractions_exceed_slots",
                                         "required": len(required_attraction_ids), "slots": len(attraction_slots)}
            return []
        # Required visits are coverage constraints in the global master, not
        # fixed day/slot assignments.  Prematurely zipping ranked entities to
        # slots prevents the optimizer from moving a must-visit attraction to
        # the day where its opening hours and route are feasible.
        self._current_required_attraction_ids = set(required_attraction_ids)

        meal_slots = [slot for template in templates for slot in template.slots if slot.kind in {"lunch", "dinner"}]
        assignments: Dict[int, int] = {}
        used_slots: set[int] = set()

        def can_assign(req_idx: int, slot_idx: int) -> bool:
            slot = meal_slots[slot_idx]
            options = restaurant_slot_sets[req_idx]
            if not options:
                return False
            template = templates[slot.day - 1]
            if template.start_time >= (slot.latest or 24 * 60) - 60:
                return False
            for rid in options:
                rec = self.state.records[rid]
                opening = rec.canonical.get("opening_time") or 0
                closing = rec.canonical.get("closing_time") or 24 * 60
                if max(opening, slot.earliest or 0, template.start_time) + 60 <= min(closing, slot.latest or 24 * 60):
                    break
            else:
                return False
            nearby = _canonical_text(restaurant_requirements[req_idx].get("nearby_location"))
            if nearby in required_names:
                matching_visits = [
                    attraction for attraction in attraction_slots
                    if any(_canonical_text(self._candidate_name(rid)) == nearby for rid in attraction.candidates)
                    and len(attraction.candidates) == 1
                ]
                if matching_visits:
                    visit = matching_visits[0]
                    if slot.day != visit.day:
                        return False
                    day_slots = templates[slot.day - 1].slots
                    if day_slots.index(slot) <= day_slots.index(visit):
                        return False
            return True

        def top_candidate_fits(req_idx: int, slot_idx: int) -> bool:
            options = restaurant_slot_sets[req_idx]
            if not options:
                return False
            slot = meal_slots[slot_idx]
            day = templates[slot.day - 1]
            rec = self.state.records[options[0]]
            opening = rec.canonical.get("opening_time") or 0
            closing = rec.canonical.get("closing_time") or 24 * 60
            return max(opening, slot.earliest or 0, day.start_time) + 60 <= min(closing, slot.latest or 24 * 60)

        def match_requirements(req_idx: int) -> bool:
            if req_idx == len(restaurant_requirements):
                return True
            slot_order = list(range(len(meal_slots)))
            if restaurant_requirements[req_idx].get("required_name"):
                slot_order.sort(key=lambda idx: (meal_slots[idx].kind != "dinner", meal_slots[idx].day))
            elif restaurant_requirements[req_idx].get("objective"):
                slot_order.sort(key=lambda idx: (not top_candidate_fits(req_idx, idx), meal_slots[idx].day,
                                                 meal_slots[idx].kind != "dinner"))
            for slot_idx in slot_order:
                if slot_idx in used_slots or not can_assign(req_idx, slot_idx):
                    continue
                assignments[req_idx] = slot_idx
                used_slots.add(slot_idx)
                if match_requirements(req_idx + 1):
                    return True
                used_slots.remove(slot_idx)
                del assignments[req_idx]
            return False

        if not match_requirements(0):
            self._template_diagnostic = {
                "reason": "restaurant_requirements_unassigned",
                "option_counts": [len(options) for options in restaurant_slot_sets],
                "meal_slots": [{"slot": slot.slot_id, "earliest": slot.earliest, "latest": slot.latest}
                               for slot in meal_slots],
            }
            return []
        for req_idx, slot_idx in assignments.items():
            slot = meal_slots[slot_idx]
            slot.candidates = restaurant_slot_sets[req_idx]
            slot.requirement_index = req_idx
        return templates

    # ---------------------------------------------------------------------
    # Route acquisition driven by current candidate graph
    # ---------------------------------------------------------------------

    def _candidate_name(self, record_id: str) -> Optional[str]:
        assert self.state is not None
        rec = self.state.records.get(record_id)
        return rec.canonical.get("name") if rec else None

    def _location_is_certified_absent(self, name: str) -> bool:
        """Whether an exact official location lookup successfully returned empty.

        This is only a candidate-level grounding failure.  It may prune a transport
        macro whose terminal cannot be connected to the itinerary, but it is never
        promoted to global problem infeasibility.
        """
        assert self.state is not None
        key = f"location:{_canonical_text(name)}"
        outcome = self.state.query_outcomes.get(key)
        return bool(
            outcome is not None
            and outcome.exhaustive
            and outcome.status == QUERY_SUCCESS_EMPTY
            and self.state.coordinates(name) is None
        )

    def _requests_for_route(self, origin_name: str, destination_name: str, purpose: str) -> List[AcquisitionRequest]:
        """Compile semantic edge -> coordinate acquisition -> official route query."""
        assert self.state is not None
        if _canonical_text(origin_name) == _canonical_text(destination_name):
            return []
        if self.state.route(origin_name, destination_name) is not None:
            return []
        requests: List[AcquisitionRequest] = []
        origin_coord = self.state.coordinates(origin_name)
        destination_coord = self.state.coordinates(destination_name)
        if origin_coord is None:
            req = self._location_request(origin_name, f"ground route origin: {purpose}")
            if self.state.can_attempt(req.key):
                requests.append(req)
        if destination_coord is None:
            req = self._location_request(destination_name, f"ground route destination: {purpose}")
            if self.state.can_attempt(req.key):
                requests.append(req)
        if requests or origin_coord is None or destination_coord is None:
            return requests
        key = f"route:{_canonical_text(origin_name)}->{_canonical_text(destination_name)}"
        if not self.state.can_attempt(key):
            return []
        return [AcquisitionRequest(
            "query_road_route_info", purpose,
            {
                "from_location": origin_name, "to_location": destination_name,
                "origin": origin_coord, "destination": destination_coord,
            },
            key, 100,
        )]

    # ---------------------------------------------------------------------
    # Candidate expansion policy after conflict
    # ---------------------------------------------------------------------

    def _expansion_requests(self, conflict: Dict[str, Any]) -> List[AcquisitionRequest]:
        assert self.state is not None
        kind = conflict.get("kind")
        reqs: List[AcquisitionRequest] = []
        if kind in {"no_restaurant_candidates", "temporal_infeasible", "coverage_infeasible", "verification_failed"}:
            for attraction in self._rank_records("attraction"):
                coordinate = _coord(attraction.canonical.get("lat"), attraction.canonical.get("lon"))
                if not coordinate:
                    continue
                key = f"restaurants:generic:{coordinate}"
                if not self.state.can_attempt(key):
                    continue
                lat, lon = coordinate.split(",", 1)
                reqs.append(AcquisitionRequest(
                    "recommend_restaurants", "expand grounded restaurant pool",
                    {"latitude": lat, "longitude": lon, "nearby_location": attraction.canonical.get("name")},
                    key, 90,
                ))
                if len(reqs) >= 3:
                    break
        return [r for r in reqs if self.tools.has(r.tool)]

    # ---------------------------------------------------------------------
    # Exact day-level temporal scheduling subproblem
    # ---------------------------------------------------------------------

    def _service_duration(self, rec: EvidenceRecord, slot_kind: str) -> int:
        if slot_kind in {"lunch", "dinner"}:
            return 60
        if slot_kind == "attraction":
            minimum = rec.canonical.get("min_duration")
            maximum = rec.canonical.get("max_duration")
            if minimum is None:
                return 120
            duration = max(30, int(minimum))
            if maximum is not None and int(maximum) > 0:
                duration = min(duration, int(maximum))
            return max(1, duration)
        return 30

    def _route_values(self, a: str, b: str) -> Tuple[Optional[int], Optional[float], Optional[EvidenceRecord]]:
        assert self.state is not None
        if _canonical_text(a) == _canonical_text(b):
            return 0, 0.0, None
        rec = self.state.route(a, b)
        if rec is None:
            return None, None, None
        duration = rec.canonical.get("duration")
        cost = rec.canonical.get("price")
        if duration is None or cost is None:
            return None, None, rec
        return int(duration), float(cost), rec

    def _route_is_certified_absent(self, origin: str, destination: str) -> bool:
        assert self.state is not None
        key = f"route:{_canonical_text(origin)}->{_canonical_text(destination)}"
        outcome = self.state.query_outcomes.get(key)
        return bool(
            outcome is not None
            and outcome.exhaustive
            and outcome.status == QUERY_SUCCESS_EMPTY
            and self.state.route(origin, destination) is None
        )

    def _schedule_day(
        self, template: DayTemplate, used_attractions: set[str], used_restaurants: set[str],
    ) -> SolveResult:
        """Earliest feasible schedule for the fixed entities/order chosen by the master."""
        assert self.state is not None
        current_name = template.start_anchor
        current_time = template.start_time
        chosen: List[Dict[str, Any]] = []
        for slot in template.slots:
            if len(slot.candidates) != 1:
                return SolveResult(
                    INFEASIBLE, "scheduler requires fixed master assignment",
                    conflict={"kind": "assignment_not_fixed", "day": template.day, "slot": slot.slot_id},
                )
            rid = slot.candidates[0]
            rec = self.state.records.get(rid)
            if rec is None or not rec.canonical.get("name"):
                return SolveResult(
                    INFEASIBLE, "selected entity lacks evidence",
                    conflict={"kind": "missing_entity", "day": template.day, "slot": slot.slot_id},
                )
            if ((slot.kind == "attraction" and rid in used_attractions)
                    or (slot.kind in {"lunch", "dinner"} and rid in used_restaurants)):
                return SolveResult(
                    INFEASIBLE, "selected entity is duplicated",
                    conflict={"kind": "duplicate_entity", "day": template.day, "slot": slot.slot_id},
                )
            travel, _, _ = self._route_values(current_name, rec.canonical["name"])
            if travel is None:
                missing = self._requests_for_route(
                    current_name, rec.canonical["name"], f"fixed day {template.day} edge",
                )
                if missing:
                    return SolveResult(NEEDS_INFO, "fixed schedule route missing", missing=missing)
                if self._route_is_certified_absent(current_name, rec.canonical["name"]):
                    return SolveResult(
                        INFEASIBLE, "fixed schedule route unavailable",
                        conflict={"kind": "route_unavailable", "day": template.day,
                                  "edge": [current_name, rec.canonical["name"]]},
                    )
                return SolveResult(
                    NEEDS_INFO, "route fact unresolved after bounded retries",
                    conflict={"kind": "route_query_unresolved", "day": template.day,
                              "edge": [current_name, rec.canonical["name"]]},
                )
            start = current_time + travel
            if slot.earliest is not None:
                start = max(start, int(slot.earliest))
            open_t = rec.canonical.get("opening_time")
            if open_t is not None:
                start = max(start, int(open_t))
            end = start + self._service_duration(rec, slot.kind)
            close_t = rec.canonical.get("closing_time")
            if ((close_t is not None and end > int(close_t))
                    or (slot.latest is not None and end > int(slot.latest))):
                return SolveResult(
                    INFEASIBLE, "fixed schedule violates time window",
                    conflict={"kind": "temporal_infeasible", "day": template.day,
                              "slot": slot.slot_id, "start": start, "end": end},
                )
            chosen.append({
                "slot_id": slot.slot_id, "kind": slot.kind, "record_id": rec.record_id,
                "name": rec.canonical["name"], "start": start, "end": end,
                "price": rec.canonical.get("price") or 0.0,
                "requirement_index": slot.requirement_index,
            })
            current_name = rec.canonical["name"]
            current_time = end
        final_travel, _, _ = self._route_values(current_name, template.end_anchor)
        if final_travel is None:
            missing = self._requests_for_route(
                current_name, template.end_anchor, f"fixed day {template.day} end edge",
            )
            if missing:
                return SolveResult(NEEDS_INFO, "fixed schedule end route missing", missing=missing)
            if self._route_is_certified_absent(current_name, template.end_anchor):
                return SolveResult(
                    INFEASIBLE, "fixed schedule end route unavailable",
                    conflict={"kind": "route_unavailable", "day": template.day,
                              "edge": [current_name, template.end_anchor]},
                )
            return SolveResult(
                NEEDS_INFO, "end-route fact unresolved after bounded retries",
                conflict={"kind": "route_query_unresolved", "day": template.day,
                          "edge": [current_name, template.end_anchor]},
            )
        deadline = max(template.end_deadline, template.start_time) if not template.slots else template.end_deadline
        if current_time + final_travel > deadline:
            return SolveResult(
                INFEASIBLE, "fixed schedule misses day deadline",
                conflict={"kind": "temporal_infeasible", "day": template.day,
                          "arrival": current_time + final_travel, "deadline": deadline},
            )
        return SolveResult(READY, solution={
            "day": template.day, "start_anchor": template.start_anchor,
            "end_anchor": template.end_anchor, "start_time": template.start_time,
            "end_deadline": deadline, "activities": chosen,
        })

    def _assemble_plan(
        self,
        outbound: EvidenceRecord,
        return_trip: EvidenceRecord,
        hotel: EvidenceRecord,
        templates: List[DayTemplate],
    ) -> SolveResult:
        assert self.state is not None
        used_attractions: set[str] = set()
        used_restaurants: set[str] = set()
        solved_days = []
        for template in templates:
            result = self._schedule_day(template, used_attractions, used_restaurants)
            if result.status != READY:
                return result
            day_solution = result.solution or {}
            solved_days.append(day_solution)
            for activity in day_solution.get("activities", []):
                if activity["kind"] == "attraction":
                    used_attractions.add(activity["record_id"])
                elif activity["kind"] in {"lunch", "dinner"}:
                    used_restaurants.add(activity["record_id"])

        solution = {
            "outbound_transport": outbound.record_id,
            "return_transport": return_trip.record_id,
            "hotel": hotel.record_id,
            "days": solved_days,
        }
        self._attach_routes_and_costs(solution)
        return SolveResult(READY, solution=solution)

    def _attach_routes_and_costs(self, solution: Dict[str, Any]) -> None:
        assert self.state is not None
        hotel = self.state.records[solution["hotel"]]
        outbound = self.state.records[solution["outbound_transport"]]
        return_trip = self.state.records[solution["return_transport"]]
        travelers = int(self.problem_ir["trip"].get("travelers", 1))
        rooms = int(self.problem_ir["trip"].get("rooms", 1))
        days = self._trip_days()
        vehicles = max(1, math.ceil(travelers / 4))

        intercity = sum(float(rec.canonical.get("price") or 0.0) * travelers for rec in (outbound, return_trip))
        hotel_cost = float(hotel.canonical.get("price") or 0.0) * rooms * max(0, days - 1)
        city_transport = 0.0
        meals = 0.0
        attractions = 0.0

        arrival_anchor = self._anchor_name(outbound, outbound=True)
        hotel_name = hotel.canonical.get("name")
        if arrival_anchor and hotel_name:
            arrival_route = self.state.route(arrival_anchor, hotel_name)
            if arrival_route:
                solution["arrival_hotel_route"] = arrival_route.record_id
                city_transport += float(arrival_route.canonical.get("price") or 0.0) * vehicles

        for day in solution["days"]:
            chain = [day["start_anchor"]] + [a["name"] for a in day.get("activities", [])] + [day["end_anchor"]]
            routes = []
            for a, b in zip(chain, chain[1:]):
                if _canonical_text(a) == _canonical_text(b):
                    continue
                rec = self.state.route(a, b)
                if rec:
                    routes.append(rec.record_id)
                    city_transport += float(rec.canonical.get("price") or 0.0) * vehicles
            day["route_records"] = routes
            for activity in day.get("activities", []):
                rec = self.state.records[activity["record_id"]]
                cost = float(rec.canonical.get("price") or 0.0) * travelers
                if activity["kind"] == "attraction":
                    attractions += cost
                elif activity["kind"] in {"lunch", "dinner"}:
                    meals += cost

        transportation = intercity + city_transport
        total = transportation + hotel_cost + meals + attractions
        solution["budget"] = {
            "transportation": transportation,
            "transportation_intercity": intercity,
            "transportation_city": city_transport,
            "accommodation": hotel_cost,
            "meals": meals,
            "attractions_and_tickets": attractions,
            "other": 0.0,
            "total_estimated_budget": total,
            "currency": self.problem_ir["trip"].get("currency") or "RMB",
        }

    # ---------------------------------------------------------------------
    # Solve -> diagnose -> acquire -> re-solve
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Feasibility-restoration helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_value_relaxable_predicate(predicate: Dict[str, Any]) -> bool:
        """Return whether a hard predicate is suitable for first-stage value relaxation.

        Stage-1 relaxation is deliberately conservative: it permits numeric/time/budget
        margins and hard arg-extreme objectives to move, but keeps entity identity,
        categorical attributes, transport mode, and named/coverage requirements fixed.
        """
        field_name = str(predicate.get("field") or "")
        op = str(predicate.get("op") or "")
        numeric_fields = {
            "price", "duration", "departure_time", "arrival_time", "rating",
            "star", "decoration_year", "distance", "route_distance", "travel_distance",
        }
        return op in {"<", "<=", ">", ">="} or field_name in numeric_fields

    def _build_relaxed_problem_ir(self, original_ir: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Create a search-only softened IR while preserving the authoritative IR.

        Levels are nested and ordered from least to most semantic relaxation:
          1 VALUE: budget, numeric/time predicates, and hard superlatives become soft.
          2 ATTRIBUTE: all entity predicates become soft; transport mode may broaden.
          3 COVERAGE: named/top-k/all-matching and restaurant anchor/name requirements
             are also released for search.  Trip endpoints/dates, official evidence,
             route grounding, temporal consistency, opening hours, and itinerary
             structure are never relaxed.
        """
        ir = copy.deepcopy(original_ir)

        def soften_spec(spec: Dict[str, Any], relax_all_predicates: bool) -> None:
            hard = list(spec.get("constraints") or [])
            kept: List[Dict[str, Any]] = []
            moved: List[Dict[str, Any]] = []
            for predicate in hard:
                if relax_all_predicates or self._is_value_relaxable_predicate(predicate):
                    relaxed = copy.deepcopy(predicate)
                    relaxed["relaxed_from_hard"] = True
                    moved.append(relaxed)
                else:
                    kept.append(predicate)
            spec["constraints"] = kept
            if moved:
                spec["soft_preferences"] = list(spec.get("soft_preferences") or []) + moved
            objective = spec.get("objective")
            if isinstance(objective, dict) and not objective.get("soft"):
                softened = copy.deepcopy(objective)
                softened["soft"] = True
                softened["relaxed_from_hard"] = True
                spec["objective"] = softened

        # Total budget is a user constraint, not a structural validity condition.
        # Remove it from the relaxed feasible set and reintroduce its violation in
        # the lexicographic restoration objective below.
        ir["trip"]["budget_max"] = None

        for direction in ("outbound", "return"):
            spec = ir["transport"][direction]
            soften_spec(spec, relax_all_predicates=(level >= 2))
            if level >= 2:
                # Mode choice is relaxable only in the broader attribute profile.
                spec["allowed_modes"] = ["flight", "train"]

        soften_spec(ir["hotel"], relax_all_predicates=(level >= 2))
        soften_spec(ir["attractions"], relax_all_predicates=(level >= 2))

        for requirement in (ir.get("restaurants") or {}).get("requirements") or []:
            soften_spec(requirement, relax_all_predicates=(level >= 2))
            if level >= 3:
                # Keep the requirement object so it can still occupy a meal slot and
                # be scored against the original requirement, but stop filtering its
                # candidate domain by identity/anchor during relaxed search.
                requirement["required_name"] = None
                requirement["nearby_location"] = None

        if level >= 3:
            attractions = ir["attractions"]
            attractions["selection"] = "default"
            attractions["count"] = None
            attractions["required_names"] = []
            attractions["selection_groups"] = []
            attractions["required_groups"] = []
            # Arbitrary global constraints are not executable in the current typed
            # layer; retain them in the authoritative IR and let final verification
            # report them as violations rather than blocking fallback construction.
            ir["global_constraints"] = []
        return ir

    def _constraint_violation_units(
        self, record: EvidenceRecord, constraints: Sequence[Dict[str, Any]],
    ) -> int:
        """Count original hard predicates not certified SAT for one grounded record."""
        return sum(self._status_against(record, [predicate]) != SAT for predicate in constraints)

    def _macro_relaxation_penalty(
        self, outbound: EvidenceRecord, inbound: EvidenceRecord, hotel: EvidenceRecord,
        fixed_cost: float,
    ) -> Tuple[int, float]:
        """Lexicographic penalty for a transport/hotel macro under the original IR."""
        original = self._relaxation_original_ir
        if not original:
            return 0, 0.0
        violations = 0
        for direction, record in (("outbound", outbound), ("return", inbound)):
            spec = original["transport"][direction]
            if record.kind not in set(spec.get("allowed_modes") or []):
                violations += 1
            violations += self._constraint_violation_units(record, spec.get("constraints") or [])
        violations += self._constraint_violation_units(hotel, (original.get("hotel") or {}).get("constraints") or [])
        budget = original["trip"].get("budget_max")
        overrun = 0.0
        if budget is not None and fixed_cost > float(budget):
            violations += 1
            overrun = (fixed_cost - float(budget)) / max(1.0, abs(float(budget)))
        return violations, overrun

    def _original_required_attraction_ids(self) -> Tuple[set[str], int]:
        """Ground original coverage requirements for restoration scoring.

        Returns (known required record ids, unresolved requirement units). Unknown
        entities/objective values are never invented; they contribute a constant
        violation unit instead of becoming fake evidence.
        """
        assert self.state is not None
        original = self._relaxation_original_ir
        if not original:
            return set(), 0
        spec = original.get("attractions") or {}
        pool = self.state.records_of("attraction")
        required: set[str] = set()
        unresolved = 0

        for name in spec.get("required_names") or []:
            matches = [r for r in pool if _canonical_text(r.canonical.get("name")) == _canonical_text(name)]
            if matches:
                required.add(matches[0].record_id)
            else:
                unresolved += 1

        selection = spec.get("selection", "default")
        if selection == "top_k":
            eligible = [r for r in pool if self._status_against(r, spec.get("constraints") or []) == SAT]
            objective = spec.get("objective") or {"field": "rating", "sense": "max"}
            scored = [(self._objective_value_for_record(r, objective), r) for r in eligible]
            known = [(value, record) for value, record in scored if value is not None]
            known.sort(key=lambda pair: float(pair[0]), reverse=str(objective.get("sense")) == "max")
            k = int(spec.get("count") or 1)
            required.update(record.record_id for _, record in known[:k])
            unresolved += max(0, k - len(known))
        elif selection == "all_matching":
            groups = spec.get("selection_groups") or []
            for record in pool:
                matches = (any(self._status_against(record, group) == SAT for group in groups)
                           if groups else self._status_against(record, spec.get("constraints") or []) == SAT)
                if matches:
                    required.add(record.record_id)

        for group in spec.get("required_groups") or []:
            eligible = [r for r in pool if self._status_against(r, group.get("constraints") or []) == SAT]
            objective = group.get("objective") or {"field": "rating", "sense": "max"}
            scored = [(self._objective_value_for_record(r, objective), r) for r in eligible]
            known = [(value, record) for value, record in scored if value is not None]
            known.sort(key=lambda pair: float(pair[0]), reverse=str(objective.get("sense")) == "max")
            count = int(group.get("count") or 1)
            required.update(record.record_id for _, record in known[:count])
            unresolved += max(0, count - len(known))
        return required, unresolved

    def _relaxation_candidate_penalty(self, slot: ActivitySlot, record: EvidenceRecord) -> int:
        """Candidate-local hard-violation count under the authoritative IR."""
        original = self._relaxation_original_ir
        if not original:
            return 0
        penalty = 0
        if slot.kind == "attraction":
            penalty += self._constraint_violation_units(
                record, (original.get("attractions") or {}).get("constraints") or [],
            )
        elif slot.kind in {"lunch", "dinner"} and slot.requirement_index is not None:
            requirements = (original.get("restaurants") or {}).get("requirements") or []
            if 0 <= slot.requirement_index < len(requirements):
                requirement = requirements[slot.requirement_index]
                required_name = requirement.get("required_name")
                if required_name and _canonical_text(record.canonical.get("name")) != _canonical_text(required_name):
                    penalty += 1
                nearby = requirement.get("nearby_location")
                if nearby and not self._restaurant_matches_anchor(record, str(nearby)):
                    penalty += 1
                penalty += self._constraint_violation_units(record, requirement.get("constraints") or [])
        return penalty

    def _verify_against_problem_ir(
        self, solution: Dict[str, Any], authoritative_ir: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify a relaxed-search candidate against the unmodified user semantics."""
        assert self.state is not None
        saved_ir = self.problem_ir
        saved_state_ir = self.state.problem_ir
        saved_engine = self.constraint_engine
        try:
            self.problem_ir = authoritative_ir
            self.state.problem_ir = authoritative_ir
            self.constraint_engine = IncrementalConstraintEngine(self)
            return self._verify(solution)
        finally:
            self.problem_ir = saved_ir
            self.state.problem_ir = saved_state_ir
            self.constraint_engine = saved_engine

    @staticmethod
    def _relaxation_violation_report(
        verification: Dict[str, Any], profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert final-verifier failures into a minimum-violation comparison key."""
        errors = list(verification.get("errors") or [])
        structural_errors: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        magnitude = 0.0
        hard_units = 0

        for error in errors:
            key = next(iter(error), "unknown")
            if key == "budget_exceeded":
                detail = error[key] or {}
                actual = _to_float(detail.get("actual"))
                maximum = _to_float(detail.get("maximum"))
                hard_units += 1
                if actual is not None and maximum is not None:
                    magnitude += max(0.0, actual - maximum) / max(1.0, abs(maximum))
                violations.append({"kind": "budget", **detail})
            elif key == "personalized_constraint_failed":
                detail = error[key] or {}
                units = max(1, len(detail.get("missing_record_ids") or []))
                hard_units += units
                selected = _to_float(detail.get("selected_value"))
                best = _to_float(detail.get("best_value"))
                if selected is not None and best is not None:
                    magnitude += abs(selected - best) / max(1.0, abs(best))
                else:
                    magnitude += float(units)
                violations.append({"kind": "personalized", **detail})
            else:
                structural_errors.append(error)

        exact = bool(verification.get("success"))
        eligible = exact or not structural_errors
        status = EXACT_FEASIBLE if exact else RELAXED_FEASIBLE if eligible else INFEASIBLE
        composite = float(verification.get("estimated_composite") or 0.0)
        total = _to_float(verification.get("budget_total")) or float("inf")
        score = (hard_units, magnitude, -composite, total)
        return {
            "status": status,
            "profile": profile.get("name"),
            "level": profile.get("level"),
            "hard_violation_count": hard_units,
            "violation_magnitude": magnitude,
            "violations": violations,
            "structural_errors": structural_errors,
            "eligible": eligible,
            "exact_under_original_ir": exact,
            "score": list(score),
        }

    def _macro_candidates(self) -> List[Tuple[EvidenceRecord, EvidenceRecord, EvidenceRecord]]:
        """Keep several grounded transport/hotel combinations, not a single pick."""
        assert self.state is not None
        trip = self.problem_ir["trip"]

        def transports(direction: str) -> List[EvidenceRecord]:
            spec = self.problem_ir["transport"][direction]
            rows = [
                rec for mode in spec["allowed_modes"]
                for rec in self._rank_records(mode, direction)
                if self._status_against(rec, spec.get("constraints") or []) == SAT
                and all(rec.canonical.get(k) is not None for k in ("price", "departure_time", "arrival_time"))
                and not self._location_is_certified_absent(
                    str(self._anchor_name(rec, outbound=(direction == "outbound")) or "")
                )
            ]
            soft = spec.get("soft_preferences") or []
            if soft and rows:
                soft_witnesses = [
                    row for row in rows
                    if all(self._status_against(row, [preference]) == SAT for preference in soft)
                ]
                if soft_witnesses:
                    rows = soft_witnesses
            objective = spec.get("objective")
            if objective and rows:
                field, sense = objective.get("field"), objective.get("sense", "min")
                rows.sort(key=lambda r: _to_float(self._record_field(r, field))
                          if _to_float(self._record_field(r, field)) is not None
                          else (float("inf") if sense == "min" else float("-inf")), reverse=sense == "max")
                if not objective.get("soft"):
                    best = _to_float(self._record_field(rows[0], field))
                    rows = [r for r in rows if _to_float(self._record_field(r, field)) == best]
                else:
                    rows.sort(key=lambda r: -sum(self._status_against(r, [pref]) == SAT for pref in soft))
            else:
                rows.sort(key=lambda r: (r.canonical.get("direct") is False,
                                         r.canonical.get("arrival_time") if direction == "outbound"
                                         else -r.canonical.get("departure_time"),
                                         r.canonical.get("price")))
            return rows[:self._pool_limit("train")]

        hotels = [r for r in self._rank_records("hotel")
                  if self._status_against(r, self.problem_ir["hotel"].get("constraints") or []) == SAT
                  and r.canonical.get("price") is not None]
        hotel_objective = self.problem_ir["hotel"].get("objective")
        if hotel_objective and hotels:
            field, sense = hotel_objective.get("field"), hotel_objective.get("sense", "min")
            hotels.sort(key=lambda r: _to_float(self._record_field(r, field))
                        if _to_float(self._record_field(r, field)) is not None
                        else (float("inf") if sense == "min" else float("-inf")), reverse=sense == "max")
            if not hotel_objective.get("soft"):
                best = _to_float(self._record_field(hotels[0], field))
                hotels = [r for r in hotels if _to_float(self._record_field(r, field)) == best]
        elif self.problem_ir["hotel"].get("soft_preferences"):
            soft = self.problem_ir["hotel"]["soft_preferences"]
            hotels.sort(key=lambda r: -sum(self._status_against(r, [pref]) == SAT for pref in soft))
        hotels = hotels[:self._pool_limit("hotel")]

        outbound, inbound = transports("outbound"), transports("return")
        if not outbound or not inbound or not hotels:
            return []
        ranked = []
        for i, j, k in itertools.product(range(len(outbound)), range(len(inbound)), range(len(hotels))):
            out, ret, hotel = outbound[i], inbound[j], hotels[k]
            fixed = (float(out.canonical["price"]) + float(ret.canonical["price"])) * int(trip["travelers"])
            fixed += float(hotel.canonical["price"]) * int(trip["rooms"]) * (self._trip_days() - 1)
            if trip.get("budget_max") is not None and fixed > trip["budget_max"]:
                continue
            violation_units, violation_magnitude = self._macro_relaxation_penalty(out, ret, hotel, fixed)
            ranked.append((
                violation_units, violation_magnitude, i + j + k, fixed,
                out.record_id, ret.record_id, hotel.record_id, out, ret, hotel,
            ))
        ranked.sort(key=lambda row: row[:7])
        return [(row[7], row[8], row[9]) for row in ranked[:24]]

    def _master_assign(
        self, templates: List[DayTemplate], macro_key: Tuple[str, str, str], fixed_cost: float,
        temporary_nogoods: Optional[set[Tuple[str, ...]]] = None,
    ) -> SolveResult:
        """Global assignment/budget master; exact route times remain a subproblem."""
        assert self.state is not None
        slots = [slot for day in templates for slot in day.slots]
        day_by_number = {day.day: day for day in templates}
        candidate_map: Dict[str, List[EvidenceRecord]] = {}
        missing_details: List[AcquisitionRequest] = []
        unresolved_detail_keys: set[str] = set()
        for slot in slots:
            records = []
            for rid in slot.candidates:
                rec = self.state.records.get(rid)
                if rec is None or not rec.canonical.get("name"):
                    continue
                if rec.canonical.get("price") is None or (slot.kind == "attraction" and rec.canonical.get("min_duration") is None):
                    tool = "query_attraction_details" if slot.kind == "attraction" else "query_restaurant_details"
                    req = AcquisitionRequest(tool, "complete candidate fact contract", {"name": rec.canonical["name"]},
                                             f"details:{rec.kind}:{_canonical_text(rec.canonical['name'])}", 80)
                    if self.state.can_attempt(req.key):
                        missing_details.append(req)
                    else:
                        outcome = self.state.query_outcomes.get(req.key)
                        if outcome is not None and outcome.status in {QUERY_ERROR, QUERY_PARSE_ERROR}:
                            unresolved_detail_keys.add(req.key)
                    continue
                attraction_spec = self.problem_ir["attractions"]
                if (slot.kind == "attraction"
                        and attraction_spec.get("selection") != "top_k"
                        and self._status_against(rec, attraction_spec.get("constraints") or []) != SAT):
                    continue
                day = day_by_number[slot.day]
                duration = self._service_duration(rec, slot.kind)
                earliest = max(day.start_time, slot.earliest or 0, rec.canonical.get("opening_time") or 0)
                latest = min(day.end_deadline, slot.latest or 24 * 60,
                             rec.canonical.get("closing_time") or 24 * 60)
                if earliest + duration > latest:
                    continue
                records.append(rec)
            if not records:
                if missing_details:
                    return SolveResult(NEEDS_INFO, "candidate facts missing", missing=missing_details)
                if unresolved_detail_keys:
                    return SolveResult(
                        COVERAGE_GAP, "candidate detail facts remain unresolved after bounded retries",
                        conflict={"kind": "candidate_detail_unresolved", "day": slot.day,
                                  "slot": slot.slot_id, "keys": sorted(unresolved_detail_keys)},
                    )
                kind = "no_attraction_candidates" if slot.kind == "attraction" else "no_restaurant_candidates"
                return SolveResult(INFEASIBLE, f"no grounded candidates for {slot.slot_id}",
                                   conflict={"kind": kind, "day": slot.day, "slot": slot.slot_id})
            candidate_map[slot.slot_id] = records
        if not HAS_GUROBI:
            return SolveResult(INFEASIBLE, "global assignment requires Gurobi",
                               conflict={"kind": "solver_unavailable"})

        started = time.time()
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("travel_global_assignment", env=env)
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 15
        x = {(slot.slot_id, rec.record_id): model.addVar(vtype=GRB.BINARY)
             for slot in slots for rec in candidate_map[slot.slot_id]}
        for slot in slots:
            model.addConstr(gp.quicksum(x[(slot.slot_id, rec.record_id)] for rec in candidate_map[slot.slot_id]) == 1)
        for rid in {rec.record_id for records in candidate_map.values() for rec in records}:
            variables = [x[(slot.slot_id, rid)] for slot in slots if (slot.slot_id, rid) in x]
            if len(variables) > 1:
                model.addConstr(gp.quicksum(variables) <= 1)
        for rid in getattr(self, "_current_required_attraction_ids", set()):
            variables = [
                x[(slot.slot_id, rid)] for slot in slots
                if slot.kind == "attraction" and (slot.slot_id, rid) in x
            ]
            if not variables:
                model.dispose()
                env.dispose()
                return SolveResult(INFEASIBLE, "required attraction has no feasible slot",
                                   conflict={"kind": "coverage_infeasible", "record_id": rid})
            model.addConstr(gp.quicksum(variables) == 1)

        def known_missing_route(origin: str, destination: str) -> bool:
            if _canonical_text(origin) == _canonical_text(destination):
                return False
            key = f"route:{_canonical_text(origin)}->{_canonical_text(destination)}"
            outcome = self.state.query_outcomes.get(key)
            return bool(
                outcome and outcome.exhaustive and outcome.status == QUERY_SUCCESS_EMPTY
                and self.state.route(origin, destination) is None
            )

        # Reuse negative route evidence at edge granularity.  Cutting only an
        # entire assignment allows the master to propose thousands of variants
        # that contain the same unavailable adjacent edge.
        for day in templates:
            if not day.slots:
                continue
            first, last = day.slots[0], day.slots[-1]
            for rec in candidate_map[first.slot_id]:
                if known_missing_route(day.start_anchor, rec.canonical["name"]):
                    model.addConstr(x[(first.slot_id, rec.record_id)] == 0)
            for rec in candidate_map[last.slot_id]:
                if known_missing_route(rec.canonical["name"], day.end_anchor):
                    model.addConstr(x[(last.slot_id, rec.record_id)] == 0)
            for left, right in zip(day.slots, day.slots[1:]):
                for a in candidate_map[left.slot_id]:
                    for b in candidate_map[right.slot_id]:
                        if known_missing_route(a.canonical["name"], b.canonical["name"]):
                            model.addConstr(
                                x[(left.slot_id, a.record_id)] + x[(right.slot_id, b.record_id)] <= 1
                            )
        travelers = int(self.problem_ir["trip"]["travelers"])
        activity_cost = gp.quicksum(float(rec.canonical["price"]) * travelers * x[(slot.slot_id, rec.record_id)]
                                    for slot in slots for rec in candidate_map[slot.slot_id])
        budget = self.problem_ir["trip"].get("budget_max")
        if budget is not None:
            # Route costs are nonnegative but not yet acquired; this is a safe
            # lower bound, checked again after the exact selected routes arrive.
            model.addConstr(fixed_cost + activity_cost <= float(budget))
        all_assignment_nogoods = set(self._assignment_nogoods.get(macro_key, set()))
        all_assignment_nogoods.update(temporary_nogoods or set())
        for assignment in all_assignment_nogoods:
            selected = [x[(slot.slot_id, rid)] for slot, rid in zip(slots, assignment)
                        if (slot.slot_id, rid) in x]
            if len(selected) == len(slots):
                model.addConstr(gp.quicksum(selected) <= len(slots) - 1)
        for day_number, assignment in self._day_nogoods.get(macro_key, set()):
            day_slots = [slot for slot in slots if slot.day == day_number]
            selected = [x[(slot.slot_id, rid)] for slot, rid in zip(day_slots, assignment)
                        if (slot.slot_id, rid) in x]
            if selected and len(selected) == len(day_slots):
                model.addConstr(gp.quicksum(selected) <= len(day_slots) - 1)
        hotel_coords = self.state.records[macro_key[2]].canonical
        hotel_lat, hotel_lon = hotel_coords.get("lat"), hotel_coords.get("lon")
        proximity_priority = gp.quicksum(
            (10.0 * (111.0 * abs(float(rec.canonical["lat"]) - float(hotel_lat))
                     + 90.0 * abs(float(rec.canonical["lon"]) - float(hotel_lon))))
            * x[(slot.slot_id, rec.record_id)]
            for slot in slots for rec in candidate_map[slot.slot_id]
            if hotel_lat is not None and hotel_lon is not None
            and rec.canonical.get("lat") is not None and rec.canonical.get("lon") is not None
        )
        # Stabilize successive masters without pair variables: prefer candidates
        # that already have at least one grounded incoming/outgoing edge to the
        # adjacent layer. This reuses official route evidence while keeping the
        # model O(sum |C_s|), instead of reintroducing O(|C_i||C_j|) binaries.
        route_reuse_terms = []
        for day in templates:
            for index, slot in enumerate(day.slots):
                left_names = (
                    [day.start_anchor] if index == 0 else
                    [record.canonical["name"] for record in candidate_map[day.slots[index - 1].slot_id]]
                )
                right_names = (
                    [day.end_anchor] if index == len(day.slots) - 1 else
                    [record.canonical["name"] for record in candidate_map[day.slots[index + 1].slot_id]]
                )
                for record in candidate_map[slot.slot_id]:
                    name = str(record.canonical["name"])
                    missing_sides = int(not any(self.state.route(left, name) for left in left_names))
                    missing_sides += int(not any(self.state.route(name, right) for right in right_names))
                    route_reuse_terms.append(
                        missing_sides * x[(slot.slot_id, record.record_id)]
                    )
        route_reuse_priority = gp.quicksum(route_reuse_terms) if route_reuse_terms else 0
        for day in templates:
            if not day.slots:
                continue
            for left, right in zip(day.slots, day.slots[1:]):
                for a in candidate_map[left.slot_id]:
                    for b in candidate_map[right.slot_id]:
                        exact = self.state.route(a.canonical["name"], b.canonical["name"])
                        if exact is not None and exact.canonical.get("duration") is not None:
                            earliest_left = max(
                                day.start_time, left.earliest or 0,
                                a.canonical.get("opening_time") or 0,
                            )
                            latest_right = min(
                                day.end_deadline, right.latest or 24 * 60,
                                b.canonical.get("closing_time") or 24 * 60,
                            )
                            required = (self._service_duration(a, left.kind)
                                        + int(exact.canonical["duration"])
                                        + self._service_duration(b, right.kind))
                            if earliest_left + required > latest_right:
                                model.addConstr(
                                    x[(left.slot_id, a.record_id)] + x[(right.slot_id, b.record_id)] <= 1
                                )
        preference_terms = []
        requirements = self._restaurant_requirements()
        for slot in slots:
            if slot.requirement_index is None or slot.requirement_index >= len(requirements):
                continue
            objective = requirements[slot.requirement_index].get("objective")
            if not isinstance(objective, dict) or not objective.get("soft"):
                continue
            values = {rec.record_id: _to_float(self._record_field(rec, objective.get("field")))
                      for rec in candidate_map[slot.slot_id]}
            known = [value for value in values.values() if value is not None]
            if not known:
                continue
            best = max(known) if objective.get("sense") == "max" else min(known)
            span = max(known) - min(known)
            scale = span if span > 1e-12 else 1.0
            for rec in candidate_map[slot.slot_id]:
                value = values[rec.record_id]
                loss = abs(best - value) / scale if value is not None else 1.0
                preference_terms.append(loss * x[(slot.slot_id, rec.record_id)])
        preference_priority = gp.quicksum(preference_terms) if preference_terms else 0

        # In feasibility-restoration mode, the objective changes from utility-first
        # search to a lexicographic minimum-violation problem. Structural constraints
        # remain hard; only explicit user constraints softened in the search IR may
        # contribute violation variables/penalties.
        relaxation_count_terms = []
        relaxation_magnitude_terms = []
        relaxation_original_ir = getattr(self, "_relaxation_original_ir", None)
        if relaxation_original_ir is not None:
            for slot in slots:
                for rec in candidate_map[slot.slot_id]:
                    units = self._relaxation_candidate_penalty(slot, rec)
                    if units:
                        relaxation_count_terms.append(units * x[(slot.slot_id, rec.record_id)])

            required_ids, unresolved_required = self._original_required_attraction_ids()
            attraction_slots = [slot for slot in slots if slot.kind == "attraction"]
            for rid in required_ids:
                vars_for_required = [
                    x[(slot.slot_id, rid)] for slot in attraction_slots
                    if (slot.slot_id, rid) in x
                ]
                if vars_for_required:
                    relaxation_count_terms.append(1 - gp.quicksum(vars_for_required))
                else:
                    relaxation_count_terms.append(1.0)
            for _ in range(unresolved_required):
                relaxation_count_terms.append(1.0)

            original_budget = relaxation_original_ir["trip"].get("budget_max")
            if original_budget is not None:
                budget_value = float(original_budget)
                max_activity = sum(
                    max(float(rec.canonical.get("price") or 0.0) * travelers
                        for rec in candidate_map[slot.slot_id])
                    for slot in slots
                ) if slots else 0.0
                max_excess = max(0.0, fixed_cost + max_activity - budget_value)
                if max_excess > 1e-9:
                    budget_slack = model.addVar(lb=0.0, ub=max_excess, name="relax_budget_slack")
                    budget_violated = model.addVar(vtype=GRB.BINARY, name="relax_budget_violated")
                    model.addConstr(budget_slack >= fixed_cost + activity_cost - budget_value)
                    model.addConstr(budget_slack <= max_excess * budget_violated)
                    relaxation_count_terms.append(budget_violated)
                    relaxation_magnitude_terms.append(
                        budget_slack / max(1.0, abs(budget_value))
                    )

        relaxation_count = gp.quicksum(relaxation_count_terms) if relaxation_count_terms else 0
        relaxation_magnitude = gp.quicksum(relaxation_magnitude_terms) if relaxation_magnitude_terms else 0

        # Keep the master linear and license-safe. Exact unknown route edges are
        # acquired only after a provisional assignment; encoding every adjacent
        # candidate pair here creates O(|C_i||C_j|) binaries and can exceed the
        # size-limited Gurobi license before any useful plan is produced.
        model.ModelSense = GRB.MINIMIZE
        if relaxation_original_ir is not None:
            model.setObjectiveN(relaxation_count, 0, priority=60, name="hard_violation_count")
            model.setObjectiveN(relaxation_magnitude, 1, priority=50, name="hard_violation_magnitude")
            model.setObjectiveN(preference_priority, 2, priority=30, name="soft_preferences")
            model.setObjectiveN(route_reuse_priority, 3, priority=25, name="route_reuse")
            model.setObjectiveN(proximity_priority, 4, priority=20, name="proximity")
            model.setObjectiveN(activity_cost, 5, priority=10, name="activity_cost")
        else:
            model.setObjectiveN(preference_priority, 0, priority=30, name="soft_preferences")
            model.setObjectiveN(route_reuse_priority, 1, priority=25, name="route_reuse")
            model.setObjectiveN(proximity_priority, 2, priority=20, name="proximity")
            model.setObjectiveN(activity_cost, 3, priority=10, name="activity_cost")
        model.optimize()
        self.metrics["solver_runtime"] += time.time() - started
        if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT} or model.SolCount == 0:
            model.dispose()
            env.dispose()
            return SolveResult(INFEASIBLE, "global candidate assignment infeasible",
                               conflict={"kind": "budget_infeasible" if budget is not None else "coverage_infeasible"})
        chosen = tuple(next(rec.record_id for rec in candidate_map[slot.slot_id]
                            if x[(slot.slot_id, rec.record_id)].X > 0.5) for slot in slots)
        self.metrics["master_assignments"] += 1
        model.dispose()
        env.dispose()
        restricted = copy.deepcopy(templates)
        for slot, rid in zip((slot for day in restricted for slot in day.slots), chosen):
            slot.candidates = [rid]
        return SolveResult(READY, solution={"templates": restricted, "assignment": chosen})

    def _selected_route_requests(self, templates: List[DayTemplate]) -> List[AcquisitionRequest]:
        assert self.state is not None
        requests = []
        for day in templates:
            chain = [day.start_anchor] + [self._candidate_name(slot.candidates[0]) for slot in day.slots] + [day.end_anchor]
            for start, end in zip(chain, chain[1:]):
                if start and end and _canonical_text(start) != _canonical_text(end):
                    requests.extend(self._requests_for_route(start, end, "selected global plan edge"))
        return list({req.key: req for req in requests}.values())

    def _attempt_solve(self) -> SolveResult:
        assert self.state is not None
        macros = self._macro_candidates()
        if not macros:
            outbound_available = any(self._rank_records(mode, "outbound") for mode in self.problem_ir["transport"]["outbound"]["allowed_modes"])
            return SolveResult(SEARCH_EXHAUSTED, "no current macro candidates",
                               conflict={"kind": "no_hotel_candidates" if outbound_available else "no_transport_candidates"})
        attraction_spec = self.problem_ir.get("attractions") or {}
        # Selection constraints describe required visits.  They must not
        # accidentally filter every ordinary sightseeing slot: e.g. requesting
        # the single highest-rated history attraction still permits other
        # attraction categories elsewhere in the itinerary.
        ranked_attractions = sorted(
            self.state.records_of("attraction"),
            key=lambda r: (
                r.canonical.get("rating") is None,
                -(r.canonical.get("rating") or 0.0),
                r.record_id,
            ),
        )
        if attraction_spec.get("selection") == "all_matching":
            groups = attraction_spec.get("selection_groups")
            attractions = ([r for r in ranked_attractions
                            if any(self._status_against(r, group) == SAT for group in groups)]
                           if groups else ranked_attractions)
        else:
            attractions = ranked_attractions[:self._pool_limit("attraction")]
        required_names = {_canonical_text(x) for x in attraction_spec.get("required_names") or []}
        present_names = {_canonical_text(r.canonical.get("name")) for r in attractions}
        if not required_names.issubset(present_names):
            return SolveResult(COVERAGE_GAP, "required attraction not in official recommendations", conflict={"kind": "no_attraction_candidates"})
        restaurants = self._rank_records("restaurant")[:self._pool_limit("restaurant")]
        if not attractions:
            return SolveResult(COVERAGE_GAP, "no attraction candidates", conflict={"kind": "no_attraction_candidates"})
        if not restaurants:
            return SolveResult(COVERAGE_GAP, "no restaurant candidates", conflict={"kind": "no_restaurant_candidates"})
        last_conflict: Dict[str, Any] = {"kind": "template_incomplete"}
        frontier_missing: Dict[str, AcquisitionRequest] = {}
        frontier_votes: Dict[str, int] = {}
        try:
            frontier_width = max(1, int(os.environ.get("TRAVEL_FRONTIER_WIDTH", "6")))
            fact_batch = max(1, int(os.environ.get("TRAVEL_FACT_BATCH", "32")))
        except ValueError:
            frontier_width = 6
            fact_batch = 32
        frontier_count = 0
        completion_requests: Optional[List[AcquisitionRequest]] = None

        def remember(requests: Sequence[AcquisitionRequest]) -> None:
            for request in requests:
                frontier_missing[request.key] = request
                frontier_votes[request.key] = frontier_votes.get(request.key, 0) + 1

        def selected_facts() -> List[AcquisitionRequest]:
            ranked = sorted(
                frontier_missing.values(),
                key=lambda request: (-frontier_votes.get(request.key, 0), -request.priority, request.key),
            )
            selected = ranked[:fact_batch]
            self.metrics["shared_fact_requests"] += len(selected)
            return selected

        for outbound, return_trip, hotel in macros:
            arrival_anchor = self._anchor_name(outbound, outbound=True)
            hotel_name = hotel.canonical.get("name")
            if not arrival_anchor or not hotel_name:
                continue
            if self.state.route(arrival_anchor, hotel_name) is None:
                requests = self._requests_for_route(
                    arrival_anchor, hotel_name, "ground arrival station/airport to hotel transfer before check-in"
                )
                if requests:
                    remember(requests)
                    frontier_count += 1
                    self.metrics["frontier_states"] += 1
                    if frontier_count >= frontier_width:
                        return SolveResult(NEEDS_INFO, "frontier macro facts required",
                                           missing=selected_facts())
                continue
            templates = self._build_templates(outbound, return_trip, hotel, attractions, restaurants)
            if not templates:
                last_conflict = {"kind": "template_incomplete", "macro": [outbound.record_id, hotel.record_id, return_trip.record_id],
                                 "details": getattr(self, "_template_diagnostic", {})}
                continue
            macro_key = (outbound.record_id, return_trip.record_id, hotel.record_id)
            fixed_cost = (float(outbound.canonical["price"]) + float(return_trip.canonical["price"])) * int(self.problem_ir["trip"]["travelers"])
            fixed_cost += float(hotel.canonical["price"]) * int(self.problem_ir["trip"]["rooms"]) * (self._trip_days() - 1)
            temporary_nogoods: set[Tuple[str, ...]] = set()
            for _ in range(8):
                master = self._master_assign(templates, macro_key, fixed_cost, temporary_nogoods)
                if master.status == NEEDS_INFO:
                    remember(master.missing)
                    return SolveResult(NEEDS_INFO, master.reason, missing=selected_facts())
                if master.status != READY or master.solution is None:
                    last_conflict = master.conflict
                    break
                selected = master.solution["templates"]
                assignment = master.solution["assignment"]
                missing = self._selected_route_requests(selected)
                if missing:
                    remember(missing)
                    unique_missing = list({request.key: request for request in missing}.values())
                    if completion_requests is None or len(unique_missing) < len(completion_requests):
                        completion_requests = unique_missing
                    temporary_nogoods.add(assignment)
                    frontier_count += 1
                    self.metrics["frontier_states"] += 1
                    if frontier_count >= frontier_width:
                        focused = completion_requests or selected_facts()
                        self.metrics["relax_and_complete_rounds"] += 1
                        self.metrics["relax_and_complete_requests"] += len(focused)
                        return SolveResult(
                            NEEDS_INFO, "relax-and-complete selected-route facts required",
                            missing=focused,
                        )
                    continue
                # A previously queried but absent edge cannot be made true by
                # another identical tool call; cut this assignment instead.
                if any(self._route_values(a, b)[0] is None
                       for day in selected
                       for a, b in zip(
                           [day.start_anchor] + [self._candidate_name(s.candidates[0]) for s in day.slots],
                           [self._candidate_name(s.candidates[0]) for s in day.slots] + [day.end_anchor]
                       ) if a and b):
                    self._assignment_nogoods.setdefault(macro_key, set()).add(assignment)
                    last_conflict = {"kind": "route_unavailable"}
                    continue
                result = self._assemble_plan(outbound, return_trip, hotel, selected)
                if result.status == READY and result.solution:
                    return result
                if result.status == NEEDS_INFO:
                    return result
                last_conflict = result.conflict
                failed_day = last_conflict.get("day") if isinstance(last_conflict, dict) else None
                if isinstance(failed_day, int):
                    day_assignment = tuple(slot.candidates[0] for day in selected if day.day == failed_day for slot in day.slots)
                    self._day_nogoods.setdefault(macro_key, set()).add((failed_day, day_assignment))
                    self.metrics["master_day_cuts"] += 1
                else:
                    self._assignment_nogoods.setdefault(macro_key, set()).add(assignment)
        if frontier_missing:
            focused = completion_requests or selected_facts()
            if completion_requests:
                self.metrics["relax_and_complete_rounds"] += 1
                self.metrics["relax_and_complete_requests"] += len(focused)
            return SolveResult(NEEDS_INFO, "frontier facts required", missing=focused)
        return SolveResult(SEARCH_EXHAUSTED, "macro/global planner exhausted candidates", conflict=last_conflict)

    # ---------------------------------------------------------------------
    # Deterministic verification
    # ---------------------------------------------------------------------

    def _objective_value_for_record(
        self, record: EvidenceRecord, objective: Dict[str, Any],
        nearby_location: Optional[str] = None,
    ) -> Optional[float]:
        assert self.state is not None
        field_name = str(objective.get("field") or "")
        if field_name in {"distance", "route_distance", "travel_distance"} and nearby_location:
            route = self.state.route(str(nearby_location), str(record.canonical.get("name") or ""))
            return _to_float(route.canonical.get("distance")) if route else None
        return _to_float(self._record_field(record, field_name))

    def _hard_objective_failure(
        self, constraint_id: str, selected: EvidenceRecord,
        domain: Sequence[EvidenceRecord], objective: Optional[Dict[str, Any]],
        nearby_location: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(objective, dict) or objective.get("soft"):
            return None
        selected_value = self._objective_value_for_record(selected, objective, nearby_location)
        values = [
            self._objective_value_for_record(record, objective, nearby_location)
            for record in domain
        ]
        if selected_value is None or not domain or any(value is None for value in values):
            return {
                "constraint_id": constraint_id,
                "reason": "hard objective is not fully grounded over the candidate domain",
                "selected": selected.canonical.get("name"),
            }
        sense = str(objective.get("sense") or "min")
        best = max(values) if sense == "max" else min(values)
        if abs(float(selected_value) - float(best)) > 1e-9:
            return {
                "constraint_id": constraint_id,
                "reason": "selected entity does not attain the requested hard objective",
                "selected": selected.canonical.get("name"),
                "selected_value": selected_value,
                "best_value": best,
            }
        return None

    def _verify_personalized_constraints(self, solution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Re-evaluate user hard requirements on the final structured solution.

        This deliberately rechecks the final plan rather than trusting candidate
        construction.  It is the internal counterpart of the official personalized
        evaluator and prevents solver/verifier semantic drift.
        """
        assert self.state is not None
        failures: List[Dict[str, Any]] = []

        def fail(constraint_id: str, reason: str, **details: Any) -> None:
            failures.append({"constraint_id": constraint_id, "reason": reason, **details})

        # Intercity transport constraints/objectives.
        for direction, solution_key in (("outbound", "outbound_transport"), ("return", "return_transport")):
            spec = self.problem_ir["transport"][direction]
            selected = self.state.records.get(solution.get(solution_key))
            if selected is None:
                fail(f"transport.{direction}.coverage", "selected transport lacks evidence")
                continue
            if selected.kind not in set(spec.get("allowed_modes") or []):
                fail(f"transport.{direction}.mode", "selected transport mode is not allowed",
                     selected_mode=selected.kind)
            status = self._status_against(selected, spec.get("constraints") or [])
            if status != SAT:
                fail(f"transport.{direction}.hard", "selected transport does not certify all hard constraints",
                     status=status, selected=selected.canonical.get("name"))
            domain = [
                record for mode in spec.get("allowed_modes") or []
                for record in self._rank_records(mode, direction)
                if self._status_against(record, spec.get("constraints") or []) == SAT
            ]
            objective_failure = self._hard_objective_failure(
                f"transport.{direction}.objective", selected, domain, spec.get("objective"),
            )
            if objective_failure:
                failures.append(objective_failure)

        # Hotel constraints/objective.
        hotel_spec = self.problem_ir.get("hotel") or {}
        hotel = self.state.records.get(solution.get("hotel"))
        if hotel is None:
            fail("hotel.coverage", "selected hotel lacks evidence")
        else:
            status = self._status_against(hotel, hotel_spec.get("constraints") or [])
            if status != SAT:
                fail("hotel.hard", "selected hotel does not certify all hard constraints",
                     status=status, selected=hotel.canonical.get("name"))
            hotel_domain = [
                record for record in self._rank_records("hotel")
                if self._status_against(record, hotel_spec.get("constraints") or []) == SAT
            ]
            objective_failure = self._hard_objective_failure(
                "hotel.objective", hotel, hotel_domain, hotel_spec.get("objective"),
            )
            if objective_failure:
                failures.append(objective_failure)

        activities = [
            activity for day in solution.get("days", [])
            for activity in day.get("activities", [])
        ]
        selected_attractions = [
            self.state.records.get(activity.get("record_id"))
            for activity in activities if activity.get("kind") == "attraction"
        ]
        selected_attractions = [record for record in selected_attractions if record is not None]
        selected_attr_ids = {record.record_id for record in selected_attractions}
        selected_attr_names = {_canonical_text(record.canonical.get("name")) for record in selected_attractions}
        attr_spec = self.problem_ir.get("attractions") or {}

        for name in attr_spec.get("required_names") or []:
            if _canonical_text(name) not in selected_attr_names:
                fail("attractions.required_names", "required attraction is absent from final itinerary", name=name)

        selection = attr_spec.get("selection", "default")
        attr_constraints = attr_spec.get("constraints") or []
        attr_pool = self.state.records_of("attraction")
        if selection == "default" and attr_constraints:
            if not any(self._status_against(record, attr_constraints) == SAT for record in selected_attractions):
                fail("attractions.hard", "no selected attraction satisfies the requested hard attraction constraints")
        elif selection == "top_k":
            eligible = [record for record in attr_pool if self._status_against(record, attr_constraints) == SAT]
            objective = attr_spec.get("objective") or {"field": "rating", "sense": "max"}
            scored = [(self._objective_value_for_record(record, objective), record) for record in eligible]
            if any(value is None for value, _ in scored):
                fail("attractions.top_k", "top-k objective values are not fully grounded")
            else:
                scored.sort(key=lambda pair: float(pair[0]), reverse=str(objective.get("sense")) == "max")
                k = int(attr_spec.get("count") or 1)
                required_ids = {record.record_id for _, record in scored[:k]}
                missing = required_ids - selected_attr_ids
                if len(scored) < k or missing:
                    fail("attractions.top_k", "final itinerary does not contain the grounded top-k attractions",
                         required=k, missing_record_ids=sorted(missing))
        elif selection == "all_matching":
            groups = attr_spec.get("selection_groups") or []
            eligible = [
                record for record in attr_pool
                if (any(self._status_against(record, group) == SAT for group in groups)
                    if groups else self._status_against(record, attr_constraints) == SAT)
            ]
            missing = {record.record_id for record in eligible} - selected_attr_ids
            if missing:
                fail("attractions.all_matching", "not every grounded matching attraction is scheduled",
                     missing_record_ids=sorted(missing))

        for group_index, group in enumerate(attr_spec.get("required_groups") or []):
            eligible = [
                record for record in attr_pool
                if self._status_against(record, group.get("constraints") or []) == SAT
            ]
            objective = group.get("objective") or {"field": "rating", "sense": "max"}
            scored = [(self._objective_value_for_record(record, objective), record) for record in eligible]
            count = int(group.get("count") or 1)
            if any(value is None for value, _ in scored) or len(scored) < count:
                fail(f"attractions.required_group.{group_index}", "required attraction group is not fully grounded")
                continue
            scored.sort(key=lambda pair: float(pair[0]), reverse=str(objective.get("sense")) == "max")
            required_ids = {record.record_id for _, record in scored[:count]}
            missing = required_ids - selected_attr_ids
            if missing:
                fail(f"attractions.required_group.{group_index}", "required attraction group is not satisfied",
                     missing_record_ids=sorted(missing))

        # Each explicit dining requirement is tied to a slot by requirement_index.
        requirements = self._restaurant_requirements()
        meals_by_requirement: Dict[int, List[EvidenceRecord]] = {}
        for activity in activities:
            req_idx = activity.get("requirement_index")
            if activity.get("kind") not in {"lunch", "dinner"} or not isinstance(req_idx, int):
                continue
            record = self.state.records.get(activity.get("record_id"))
            if record is not None:
                meals_by_requirement.setdefault(req_idx, []).append(record)
        for index, requirement in enumerate(requirements):
            selected_records = meals_by_requirement.get(index, [])
            if not selected_records:
                fail(f"restaurants.requirement.{index}", "no final meal is assigned to this dining requirement")
                continue
            selected = selected_records[0]
            required_name = requirement.get("required_name")
            nearby = requirement.get("nearby_location")
            if required_name and _canonical_text(selected.canonical.get("name")) != _canonical_text(required_name):
                fail(f"restaurants.requirement.{index}.name", "wrong named restaurant selected",
                     selected=selected.canonical.get("name"), required=required_name)
            if nearby and not self._restaurant_matches_anchor(selected, str(nearby)):
                fail(f"restaurants.requirement.{index}.nearby", "restaurant is not grounded to the required nearby location",
                     selected=selected.canonical.get("name"), required=nearby)
            status = self._status_against(selected, requirement.get("constraints") or [])
            if status != SAT:
                fail(f"restaurants.requirement.{index}.hard", "selected restaurant does not certify all hard constraints",
                     status=status, selected=selected.canonical.get("name"))
            base_requirement = copy.deepcopy(requirement)
            base_requirement["objective"] = None
            domain = self._restaurants_for_requirement(base_requirement)
            objective_failure = self._hard_objective_failure(
                f"restaurants.requirement.{index}.objective", selected, domain,
                requirement.get("objective"), nearby_location=nearby,
            )
            if objective_failure:
                failures.append(objective_failure)

        if self.problem_ir.get("global_constraints"):
            # The current public IR schema does not define executable semantics for
            # arbitrary global constraints. Fail closed instead of silently passing.
            fail("global_constraints", "non-empty global constraints are not implemented by the typed verifier")
        return failures

    def _recompute_budget(self, solution: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Independent deterministic cost recomputation used only by verification."""
        assert self.state is not None
        outbound = self.state.records.get(solution.get("outbound_transport"))
        inbound = self.state.records.get(solution.get("return_transport"))
        hotel = self.state.records.get(solution.get("hotel"))
        if not outbound or not inbound or not hotel:
            return None
        travelers = int(self.problem_ir["trip"].get("travelers", 1))
        rooms = int(self.problem_ir["trip"].get("rooms", 1))
        vehicles = max(1, math.ceil(travelers / 4))
        try:
            intercity = sum(float(record.canonical["price"]) * travelers for record in (outbound, inbound))
            accommodation = float(hotel.canonical["price"]) * rooms * max(0, self._trip_days() - 1)
        except (KeyError, TypeError, ValueError):
            return None
        city_transport = 0.0
        arrival_anchor = self._anchor_name(outbound, outbound=True)
        hotel_name = hotel.canonical.get("name")
        if arrival_anchor and hotel_name and _canonical_text(arrival_anchor) != _canonical_text(hotel_name):
            route = self.state.route(arrival_anchor, hotel_name)
            if route is None or route.canonical.get("price") is None:
                return None
            city_transport += float(route.canonical["price"]) * vehicles
        meals = attractions = 0.0
        for day in solution.get("days", []):
            chain = [day["start_anchor"]] + [a["name"] for a in day.get("activities", [])] + [day["end_anchor"]]
            for a, b in zip(chain, chain[1:]):
                if _canonical_text(a) == _canonical_text(b):
                    continue
                route = self.state.route(a, b)
                if route is None or route.canonical.get("price") is None:
                    return None
                city_transport += float(route.canonical["price"]) * vehicles
            for activity in day.get("activities", []):
                record = self.state.records.get(activity.get("record_id"))
                if record is None or record.canonical.get("price") is None:
                    return None
                cost = float(record.canonical["price"]) * travelers
                if activity.get("kind") == "attraction":
                    attractions += cost
                elif activity.get("kind") in {"lunch", "dinner"}:
                    meals += cost
        transportation = intercity + city_transport
        total = transportation + accommodation + meals + attractions
        return {
            "transportation": transportation,
            "transportation_intercity": intercity,
            "transportation_city": city_transport,
            "accommodation": accommodation,
            "meals": meals,
            "attractions_and_tickets": attractions,
            "other": 0.0,
            "total_estimated_budget": total,
        }

    def _verify(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        assert self.state is not None
        errors: List[Dict[str, Any]] = []
        used_attr: set[str] = set()
        used_rest: set[str] = set()

        outbound = self.state.records.get(solution.get("outbound_transport"))
        hotel = self.state.records.get(solution.get("hotel"))
        if outbound and hotel:
            arrival_anchor = self._anchor_name(outbound, outbound=True)
            hotel_name = hotel.canonical.get("name")
            if arrival_anchor and hotel_name:
                required_pre_route = self.state.route(arrival_anchor, hotel_name)
                if required_pre_route is None:
                    errors.append({"missing_arrival_hotel_route": [arrival_anchor, hotel_name]})

        for day in solution.get("days", []):
            activities = day.get("activities", [])
            chain_names = [day["start_anchor"]] + [a["name"] for a in activities] + [day["end_anchor"]]
            # Every location transition must have route evidence.
            for a, b in zip(chain_names, chain_names[1:]):
                if _canonical_text(a) == _canonical_text(b):
                    continue
                route = self.state.route(a, b)
                if route is None:
                    errors.append({"missing_route_evidence": [a, b], "day": day["day"]})

            previous_end = day["start_time"]
            previous_name = day["start_anchor"]
            for activity in activities:
                rec = self.state.records.get(activity["record_id"])
                if rec is None:
                    errors.append({"missing_entity_evidence": activity["record_id"]})
                    continue
                travel, _, _ = self._route_values(previous_name, activity["name"])
                if travel is None:
                    errors.append({"missing_route_fact": [previous_name, activity["name"]], "day": day["day"]})
                elif activity["start"] < previous_end + travel:
                    errors.append({"temporal_overlap_or_missing_travel": activity["slot_id"], "day": day["day"]})
                if activity["end"] <= activity["start"]:
                    errors.append({"nonpositive_duration": activity["slot_id"]})
                open_t = rec.canonical.get("opening_time")
                close_t = rec.canonical.get("closing_time")
                if open_t is not None and activity["start"] < open_t:
                    errors.append({"before_open": activity["name"], "day": day["day"]})
                if close_t is not None and activity["end"] > close_t:
                    errors.append({"after_close": activity["name"], "day": day["day"]})
                if activity["kind"] in {"lunch", "dinner"}:
                    dur = activity["end"] - activity["start"]
                    if dur < 60 or dur > 120:
                        errors.append({"meal_duration": activity["name"], "minutes": dur})
                if activity["kind"] == "attraction":
                    if activity["record_id"] in used_attr:
                        errors.append({"duplicate_attraction": activity["name"]})
                    used_attr.add(activity["record_id"])
                    min_dur = rec.canonical.get("min_duration")
                    max_dur = rec.canonical.get("max_duration")
                    dur = activity["end"] - activity["start"]
                    if min_dur is not None and dur < min_dur:
                        errors.append({"attraction_too_short": activity["name"], "minutes": dur})
                    if max_dur is not None and dur > max_dur:
                        errors.append({"attraction_too_long": activity["name"], "minutes": dur})
                if activity["kind"] in {"lunch", "dinner"}:
                    if activity["record_id"] in used_rest:
                        errors.append({"duplicate_restaurant": activity["name"]})
                    used_rest.add(activity["record_id"])
                previous_end = activity["end"]
                previous_name = activity["name"]

            final_travel, _, _ = self._route_values(previous_name, day["end_anchor"])
            if final_travel is None:
                errors.append({"missing_route_fact": [previous_name, day["end_anchor"]], "day": day["day"]})
            elif previous_end + final_travel > day["end_deadline"]:
                errors.append({"day_end_deadline_missed": day["day"]})

            attraction_count = sum(activity["kind"] == "attraction" for activity in activities)
            if day["day"] == 1 and outbound and outbound.canonical.get("arrival_time", 24 * 60) < 12 * 60 and attraction_count < 1:
                errors.append({"arrival_day_attraction_coverage": day["day"]})
            elif day["day"] == self._trip_days():
                inbound = self.state.records.get(solution.get("return_transport"))
                if inbound and inbound.canonical.get("departure_time", 0) > RETURN_DAY_ATTRACTION_THRESHOLD and attraction_count < 1:
                    errors.append({"return_day_attraction_coverage": day["day"]})
            elif 1 < day["day"] < self._trip_days() and attraction_count < 2:
                errors.append({"full_day_attraction_coverage": day["day"]})

        budget_max = self.problem_ir["trip"].get("budget_max")
        total = (solution.get("budget") or {}).get("total_estimated_budget")
        if budget_max is not None and total is not None and float(total) > float(budget_max) + 1e-6:
            errors.append({"budget_exceeded": {"actual": total, "maximum": budget_max}})

        # Independent cost recomputation: rendered/attached budget must agree with
        # the grounded entities and route evidence used by the final plan.
        recomputed_budget = self._recompute_budget(solution)
        if recomputed_budget is None:
            errors.append({"cost_calculation_unverifiable": True})
        else:
            attached_budget = solution.get("budget") or {}
            mismatches = {}
            for key, expected in recomputed_budget.items():
                actual = _to_float(attached_budget.get(key))
                if actual is None or abs(float(actual) - float(expected)) > 1e-6:
                    mismatches[key] = {"actual": actual, "expected": expected}
            if mismatches:
                errors.append({"cost_calculation_mismatch": mismatches})

        personalized_failures = self._verify_personalized_constraints(solution)
        for failure in personalized_failures:
            errors.append({"personalized_constraint_failed": failure})

        # Evidence coverage: required entities + route edges are all grounded.
        required = 0
        supported = 0
        for day in solution.get("days", []):
            for activity in day.get("activities", []):
                required += 1
                if activity["record_id"] in self.state.records:
                    supported += 1
            chain = [day["start_anchor"]] + [a["name"] for a in day.get("activities", [])] + [day["end_anchor"]]
            for a, b in zip(chain, chain[1:]):
                if _canonical_text(a) == _canonical_text(b):
                    continue
                required += 1
                if self.state.route(a, b):
                    supported += 1
        coverage = supported / required if required else 1.0
        dimension_error_keys = {
            "route_consistency": {
                "missing_arrival_hotel_route", "missing_route_evidence", "missing_route_fact",
            },
            "sandbox_compliance": {"missing_entity_evidence"},
            "itinerary_structure": {"arrival_day_attraction_coverage", "return_day_attraction_coverage",
                                    "full_day_attraction_coverage"},
            "time_feasibility": {"temporal_overlap_or_missing_travel", "day_end_deadline_missed"},
            "business_hours": {"before_open", "after_close"},
            "duration_rationality": {"nonpositive_duration", "meal_duration",
                                     "attraction_too_short", "attraction_too_long"},
            "cost_calculation_accuracy": {"cost_calculation_unverifiable", "cost_calculation_mismatch"},
            "activity_diversity": {"duplicate_attraction", "duplicate_restaurant"},
        }
        observed_error_keys = {next(iter(error), "unknown") for error in errors}
        dimension_scores = {
            name: float(not bool(keys & observed_error_keys))
            for name, keys in dimension_error_keys.items()
        }
        estimated_commonsense = sum(dimension_scores.values()) / len(dimension_scores)
        estimated_personalized = float(not personalized_failures and "budget_exceeded" not in observed_error_keys)
        estimated_composite = (estimated_commonsense + estimated_personalized) / 2.0
        return {
            "success": not errors,
            "errors": errors,
            "evidence_coverage": coverage,
            "budget_total": total,
            "budget_max": budget_max,
            "recomputed_budget": recomputed_budget,
            "personalized_failures": personalized_failures,
            "commonsense_dimensions": dimension_scores,
            "estimated_commonsense": estimated_commonsense,
            "estimated_personalized": estimated_personalized,
            "estimated_composite": estimated_composite,
        }

    # ---------------------------------------------------------------------
    # Deterministic rendering from verified structured solution
    # ---------------------------------------------------------------------

    def _render_plan(self, converted: Dict[str, Any]) -> str:
        """Render a benchmark JSON artifact as a human-readable view."""
        lines = ["<plan>"]
        for day in converted["daily_plans"]:
            lines.append(f"Day {day['day_number']}:")
            lines.append(f"Current City: {day['current_city']}")
            accommodation = day.get("accommodation")
            lines.append(
                f"Accommodation: {accommodation['name']}, {accommodation['price_per_night']}/room/night"
                if accommodation else "Accommodation: -"
            )
            for activity in day["activities"]:
                kind, details = activity["type"], activity["details"]
                if kind == "travel_intercity_public":
                    value = (f"{details['mode']} {details['number']}, {details['from']} - "
                             f"{details['to']}, {details['cost']}/person")
                elif kind == "travel_city":
                    value = (f"{details['from']} - {details['to']}, {details['distance']}, "
                             f"{details['duration']}, {details['cost']}")
                elif kind == "hotel":
                    value = f"{details['activity']}, {details['name']}"
                elif kind == "attraction":
                    value = f"{details['name']}, {details['cost']}/person"
                elif kind == "meal":
                    meal = "Lunch" if details["meal_type"] == "午餐" else "Dinner"
                    value = f"{meal}, {details['name']}, {details['cost']}/person"
                else:
                    value = details.get("description", "")
                lines.append(f"{activity['time_slot']} | {kind} | {value}")
            lines.append("")
        budget = converted["budget_summary"]
        lines.append("**Budget Summary**:")
        for label, field in (
            ("Transportation", "transportation"),
            ("Accommodation", "accommodation"),
            ("Meals", "meals"),
            ("Attractions & Tickets", "attractions_and_tickets"),
            ("Other", "other"),
            ("Total Estimated Budget", "total_estimated_budget"),
        ):
            lines.append(f"   **{label}: {float(budget.get(field, 0)):.2f} {budget['currency']}**")
        lines.append("</plan>")
        return "\n".join(lines)

    def to_converted_plan(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Build benchmark JSON directly from verified records and solver times."""
        assert self.state is not None
        outbound = self.state.records[solution["outbound_transport"]]
        inbound = self.state.records[solution["return_transport"]]
        hotel = self.state.records[solution["hotel"]]
        trip = self.problem_ir["trip"]
        days: List[Dict[str, Any]] = []

        def add(day: Dict[str, Any], start: int, end: int, kind: str, details: Dict[str, Any]) -> None:
            day["activities"].append({
                "time_slot": f"{_fmt_hhmm(start)}-{_fmt_hhmm(end)}",
                "type": kind,
                "details": details,
            })

        def city_route(day: Dict[str, Any], start: int, origin: str, destination: str) -> int:
            if _canonical_text(origin) == _canonical_text(destination):
                return start
            route = self.state.route(origin, destination)
            if route is None or route.canonical.get("duration") is None or route.canonical.get("price") is None:
                raise ValueError(f"unverified route {origin} -> {destination}")
            c = route.canonical
            duration = int(c["duration"])
            distance = c.get("distance")
            if isinstance(distance, (int, float)):
                distance = f"{float(distance) / 1000:.1f}km"
            add(day, start, start + duration, "travel_city", {
                "from": origin, "to": destination, "distance": distance or "",
                "duration": f"{duration}min", "cost": float(c["price"]),
            })
            return start + duration

        def intercity(day: Dict[str, Any], rec: EvidenceRecord) -> None:
            c = rec.canonical
            add(day, int(c["departure_time"]), int(c["arrival_time"]), "travel_intercity_public", {
                "mode": c["mode"], "number": c["name"],
                "from": c.get("departure_station") or c.get("origin"),
                "to": c.get("arrival_station") or c.get("destination"),
                "cost": float(c["price"]),
            })

        for solved in solution["days"]:
            number = solved["day"]
            if number == 1:
                city = f"from {trip['origin']} to {trip['destination']}"
            elif number == self._trip_days():
                city = f"from {trip['destination']} to {trip['origin']}"
            else:
                city = trip["destination"]
            day: Dict[str, Any] = {"day_number": number, "current_city": city, "activities": []}
            if number < self._trip_days():
                day["accommodation"] = {
                    "name": hotel.canonical["name"],
                    "price_per_night": float(hotel.canonical["price"]),
                }
            days.append(day)
            if number == 1:
                intercity(day, outbound)
                arrival = int(outbound.canonical["arrival_time"])
                add(day, arrival, arrival + 45, "buffer", {"description": "Arrival procedures"})
                anchor = self._anchor_name(outbound, outbound=True)
                checkin = city_route(day, arrival + 45, anchor, hotel.canonical["name"])
                add(day, checkin, checkin + 30, "hotel", {
                    "activity": "Check-in", "name": hotel.canonical["name"],
                })

            current_name = solved["start_anchor"]
            current_time = int(solved["start_time"])
            for activity in solved.get("activities", []):
                current_time = city_route(day, current_time, current_name, activity["name"])
                start, end = int(activity["start"]), int(activity["end"])
                if current_time < start:
                    add(day, current_time, start, "buffer", {"description": "Waiting / rest"})
                rec = self.state.records[activity["record_id"]]
                if activity["kind"] == "attraction":
                    details = {"name": activity["name"], "city": trip["destination"],
                               "cost": float(rec.canonical["price"])}
                    kind = "attraction"
                else:
                    details = {"meal_type": "午餐" if activity["kind"] == "lunch" else "晚餐",
                               "name": activity["name"], "cost": float(rec.canonical["price"])}
                    kind = "meal"
                add(day, start, end, kind, details)
                current_name, current_time = activity["name"], end
            current_time = city_route(day, current_time, current_name, solved["end_anchor"])
            if number < self._trip_days():
                if current_time < 23 * 60:
                    add(day, current_time, 23 * 60, "hotel", {
                        "activity": "Rest", "name": hotel.canonical["name"],
                    })
            else:
                departure = int(inbound.canonical["departure_time"])
                if current_time < departure:
                    add(day, current_time, departure, "buffer", {
                        "description": "Security check / waiting for boarding",
                    })
                intercity(day, inbound)
        budget = dict(solution["budget"])
        budget["currency"] = "CNY"
        return {"budget_summary": budget, "daily_plans": days}

    # ---------------------------------------------------------------------
    # Public run method
    # ---------------------------------------------------------------------

    def _run_acquisition_queue(
        self, requests: Iterable[AcquisitionRequest], include_followups: bool = False,
    ) -> bool:
        """Execute every request source through one priority/dedup queue."""
        assert self.state is not None
        pending = {
            request.key: request for request in requests
            if self.state.can_attempt(request.key)
        }
        scheduled = set(pending)
        made_progress = False
        while pending and self.metrics["tool_calls"] < self.max_tool_calls:
            request = max(pending.values(), key=lambda item: (item.priority, item.key))
            pending.pop(request.key, None)
            made_progress |= self._execute_request(request)
            if not include_followups:
                continue
            followups: Iterable[AcquisitionRequest] = ()
            if request.tool == "recommend_attractions":
                followups = itertools.chain(self._detail_requests(), self._restaurant_requests())
            elif request.tool in {"query_attraction_details", "search_location"}:
                followups = self._restaurant_requests()
            elif request.tool == "recommend_restaurants":
                followups = self._detail_requests()
            for followup in followups:
                if followup.key in scheduled or not self.state.can_attempt(followup.key):
                    continue
                scheduled.add(followup.key)
                pending[followup.key] = followup
        return made_progress

    def _propagate_constraints(self, max_passes: int = 4) -> Optional[SolveResult]:
        """Run symbolic fast/lack checks before paying for selected routes."""
        assert self.state is not None and self.constraint_engine is not None
        for _ in range(max_passes):
            report = self.constraint_engine.propagate()
            self.last_propagation = report
            self.metrics["constraint_propagations"] += 1
            self.metrics["constraint_false"] += sum(
                item.constraint.hard and item.truth == UNSAT for item in report.evaluations
            )
            self.metrics["constraint_unknown"] += report.unresolved
            if report.truth == UNSAT:
                certificate = next((
                    item.certificate for item in report.evaluations
                    if item.constraint.hard and item.truth == UNSAT
                    and item.certificate is not None
                    and self._verify_infeasibility_certificate(item.certificate)
                ), None)
                if certificate is not None:
                    self.metrics["infeasibility_certificates_issued"] += 1
                    return SolveResult(
                        PROVEN_INFEASIBLE, "certified symbolic infeasibility",
                        conflict={"kind": "certified_infeasible", "constraints": report.conflicts},
                        certificate=certificate,
                    )
                self.metrics["infeasibility_certificates_rejected"] += 1
                return SolveResult(
                    COVERAGE_GAP, "uncertified symbolic conflict",
                    conflict={"kind": "uncertified_symbolic_conflict", "constraints": report.conflicts},
                )
            if not report.missing:
                return None
            self.metrics["fastcheck_fact_requests"] += len(report.missing)
            made_progress = self._run_acquisition_queue(report.missing)
            if not made_progress:
                return None
        return None

    def _verify_infeasibility_certificate(
        self, certificate: InfeasibilityCertificate,
    ) -> bool:
        """Fail closed: only deterministic, exhaustive evidence can certify infeasibility."""
        if (self.state is None or self.constraint_engine is None
                or not certificate.evidence_keys or not certificate.constraints):
            return False
        outcomes = [self.state.query_outcomes.get(key) for key in certificate.evidence_keys]
        if not all(outcome is not None and outcome.exhaustive
                   and outcome.status in {QUERY_SUCCESS_EMPTY, QUERY_SUCCESS_NONEMPTY}
                   for outcome in outcomes):
            return False
        constraints = {
            item.constraint_id: item for item in self.constraint_engine.constraints
        }

        def semantically_uncertain(constraint: ConstraintIR) -> bool:
            source = _canonical_text(constraint.source_text)
            query = _canonical_text(getattr(self, "current_user_query", ""))
            source_context = source
            if source and query:
                position = query.find(source)
                if position >= 0:
                    source_context = query[max(0, position - 12):position + len(source)]
            return bool(source_context and any(marker in source_context for marker in (
                "非必须", "不强求", "可有可无", "如果有更好", "optional", "nice to have",
            )))

        for constraint_id in certificate.constraints:
            constraint = constraints.get(constraint_id)
            if constraint is None or not constraint.hard:
                return False
            # A proof is only as sound as the compiled modality. If a hard
            # predicate cites explicitly preferential language, fail closed and
            # send it through semantic reflection instead of certifying the
            # natural-language request as infeasible.
            if semantically_uncertain(constraint):
                return False
            if (certificate.kind == "EXHAUSTIVE_NO_WITNESS"
                    and constraint.kind == "candidate_coverage"
                    and any(
                        item.hard and item.kind == "predicate"
                        and item.entity == constraint.entity
                        and semantically_uncertain(item)
                        for item in constraints.values()
                    )):
                return False
            current = self.constraint_engine._evaluate(constraint)
            if current.truth != UNSAT or current.certificate != certificate:
                return False
        if certificate.kind == "BUDGET_LOWER_BOUND":
            return (certificate.lhs is not None and certificate.rhs is not None
                    and certificate.lhs > certificate.rhs)
        if certificate.kind == "EXACT_ENTITY_ABSENCE":
            return all(outcome.status == QUERY_SUCCESS_EMPTY for outcome in outcomes if outcome)
        if certificate.kind == "EXHAUSTIVE_NO_WITNESS":
            return True
        return False

    def _attempt_feasibility_restoration(
        self,
        authoritative_ir: Dict[str, Any],
        trigger_result: Optional[SolveResult],
        existing_incumbent: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Search for the closest structurally valid plan after exact recovery stalls.

        Reflection is responsible for semantic/search-space repair.  This phase does
        not ask the LLM to weaken the request.  Instead, it deterministically solves
        progressively softened search models and evaluates every candidate against
        the unchanged authoritative IR.  The first relaxation profile that yields a
        structurally valid candidate wins; within that profile candidates are ordered
        by (hard violation count, violation magnitude, -estimated composite, cost).
        """
        if not getattr(self, "enable_relaxation", False) or self.state is None or not HAS_GUROBI:
            return None

        self.metrics["relaxation_attempted"] = True
        trigger_status = trigger_result.status if trigger_result else None
        trigger_reason = trigger_result.reason if trigger_result else None
        profiles = [
            {
                "name": "VALUE_RELAXATION", "level": 1,
                "description": "relax budget/numeric margins and hard superlatives",
            },
            {
                "name": "ATTRIBUTE_RELAXATION", "level": 2,
                "description": "add categorical predicates and transport mode",
            },
            {
                "name": "COVERAGE_RELAXATION", "level": 3,
                "description": "add named/top-k/all-matching and restaurant anchor/name coverage",
            },
        ]

        # Preserve the exact-search controller state.  Relaxed assignments must not
        # pollute exact no-good sets, while newly acquired official evidence is kept.
        saved_ir = self.problem_ir
        saved_state_ir = self.state.problem_ir
        saved_engine = self.constraint_engine
        saved_assignment_nogoods = self._assignment_nogoods
        saved_day_nogoods = self._day_nogoods
        saved_required_ids = set(getattr(self, "_current_required_attraction_ids", set()))
        saved_profile = self._relaxation_profile
        saved_original = self._relaxation_original_ir

        fallback_candidate: Optional[Dict[str, Any]] = None
        if existing_incumbent is not None and existing_incumbent.get("solution"):
            verification = self._verify_against_problem_ir(
                existing_incumbent["solution"], authoritative_ir,
            )
            report = self._relaxation_violation_report(
                verification, {"name": "EXACT_SEARCH_INCUMBENT", "level": 0},
            )
            if report["eligible"]:
                fallback_candidate = {
                    "solution": copy.deepcopy(existing_incumbent["solution"]),
                    "verification": verification,
                    "report": report,
                }

        try:
            self._relaxation_original_ir = authoritative_ir
            for profile in profiles:
                relaxed_ir = self._build_relaxed_problem_ir(authoritative_ir, int(profile["level"]))
                self._relaxation_profile = profile
                self.problem_ir = relaxed_ir
                self.state.problem_ir = relaxed_ir
                self.constraint_engine = IncrementalConstraintEngine(self)
                self._assignment_nogoods = {}
                self._day_nogoods = {}
                self._current_required_attraction_ids = set()
                self.metrics["relaxation_profiles_attempted"] += 1

                # A broader relaxed domain may require transport/hotel/attraction
                # calls that were intentionally not made under the exact IR.
                if self.metrics["tool_calls"] < self.max_tool_calls:
                    try:
                        self._run_acquisition_queue(
                            self._initial_requests(), include_followups=True,
                        )
                    except RuntimeError as error:
                        if "tool-call budget exhausted" not in str(error):
                            raise

                profile_best: Optional[Dict[str, Any]] = None
                candidates_seen = 0
                for _ in range(self.max_relaxation_rounds):
                    self.metrics["relaxation_solver_rounds"] += 1
                    result = self._attempt_solve()

                    if result.status == NEEDS_INFO:
                        if self.metrics["tool_calls"] >= self.max_tool_calls:
                            break
                        try:
                            if self._run_acquisition_queue(result.missing, include_followups=False):
                                continue
                        except RuntimeError as error:
                            if "tool-call budget exhausted" not in str(error):
                                raise
                        break

                    if result.status == READY and result.solution:
                        verification = self._verify_against_problem_ir(
                            result.solution, authoritative_ir,
                        )
                        report = self._relaxation_violation_report(verification, profile)
                        report.update({
                            "trigger_status": trigger_status,
                            "trigger_reason": trigger_reason,
                            "description": profile.get("description"),
                        })
                        self.metrics["relaxation_candidates"] += 1
                        candidates_seen += 1

                        if report["eligible"]:
                            candidate = {
                                "solution": copy.deepcopy(result.solution),
                                "verification": copy.deepcopy(verification),
                                "report": copy.deepcopy(report),
                            }
                            if (profile_best is None
                                    or tuple(report["score"]) < tuple(profile_best["report"]["score"])):
                                profile_best = candidate
                            if report["exact_under_original_ir"]:
                                self.metrics["relaxation_exact_recovered"] = True
                                return candidate

                        # Force a different assignment on the next relaxed solve.
                        macro_key = (
                            result.solution.get("outbound_transport"),
                            result.solution.get("return_transport"),
                            result.solution.get("hotel"),
                        )
                        assignment = tuple(
                            activity["record_id"]
                            for day in result.solution.get("days", [])
                            for activity in day.get("activities", [])
                            if activity.get("record_id")
                        )
                        if all(macro_key) and assignment:
                            self._assignment_nogoods.setdefault(macro_key, set()).add(assignment)
                        if candidates_seen >= self.max_relaxed_candidates:
                            break
                        continue

                    if result.status in {INFEASIBLE, COVERAGE_GAP, SEARCH_EXHAUSTED}:
                        requests = self._expansion_requests(result.conflict or {})
                        if requests and self.metrics["tool_calls"] < self.max_tool_calls:
                            try:
                                if self._run_acquisition_queue(requests, include_followups=True):
                                    continue
                            except RuntimeError as error:
                                if "tool-call budget exhausted" not in str(error):
                                    raise
                        break

                    # A certificate is meaningful for the exact problem, but a
                    # relaxed search profile may still construct a best-effort plan.
                    # There is no additional action for other terminal statuses.
                    break

                # Profile ordering is itself lexicographic: never move to a more
                # permissive semantic relaxation if a structurally valid candidate
                # exists at the current, less severe level.
                if profile_best is not None:
                    return profile_best

            return fallback_candidate
        finally:
            self.problem_ir = saved_ir
            self.state.problem_ir = saved_state_ir
            self.constraint_engine = saved_engine
            self._assignment_nogoods = saved_assignment_nogoods
            self._day_nogoods = saved_day_nogoods
            self._current_required_attraction_ids = saved_required_ids
            self._relaxation_profile = saved_profile
            self._relaxation_original_ir = saved_original

    def run(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        self._reset_metrics()
        self.current_user_query = user_query
        self.semantic_events = []
        self._reflection_signatures = set()
        self.relaxation_report = None
        started = time.time()
        best_incumbent: Optional[Dict[str, Any]] = None
        try:
            self.problem_ir = self._parse_problem_ir(user_query)
            self.problem_ir = self._audit_problem_ir(user_query, self.problem_ir)
            self.state = TravelState(self.problem_ir)
            self.constraint_engine = IncrementalConstraintEngine(self)

            self._run_acquisition_queue(
                self._initial_requests(), include_followups=True,
            )

            # 3) Closed loop: solve -> missing info / conflict -> acquire -> re-solve.
            last_result: Optional[SolveResult] = None
            for round_idx in range(1, self.max_solver_rounds + 1):
                self.metrics["solver_rounds"] = round_idx
                propagated = self._propagate_constraints()
                # Propagation can return either a formally certified terminal
                # infeasibility or a recoverable COVERAGE_GAP.  Do not terminate
                # merely because propagation returned a SolveResult: recoverable
                # results must flow through the same expansion/reflection controller
                # as solver-side failures.
                result = propagated if propagated is not None else self._attempt_solve()
                last_result = result

                if result.status == READY and result.solution:
                    verification = self._verify(result.solution)
                    incumbent_score = float(verification.get("estimated_composite") or 0.0)
                    if best_incumbent is None or incumbent_score > best_incumbent["score"]:
                        best_incumbent = {
                            "score": incumbent_score,
                            "solution": copy.deepcopy(result.solution),
                            "verification": copy.deepcopy(verification),
                        }
                        self.metrics["incumbent_updates"] += 1
                        self.state.solution = best_incumbent["solution"]
                        self.state.verification = best_incumbent["verification"]
                    self.metrics["verification_success"] = bool(verification["success"])
                    self.metrics["evidence_coverage"] = verification["evidence_coverage"]
                    if verification["success"]:
                        self.metrics["reflection_recovery_success"] = self.metrics["reflection_actions_accepted"] > 0
                        converted = self.to_converted_plan(result.solution)
                        self.state.converted_plan = converted
                        plan = self._render_plan(converted)
                        self.metrics["elapsed_time"] = time.time() - started
                        return plan, self._diagnostic_payload()
                    macro_key = (
                        result.solution["outbound_transport"],
                        result.solution["return_transport"],
                        result.solution["hotel"],
                    )
                    failed_assignment = tuple(
                        activity["record_id"]
                        for day in result.solution.get("days", [])
                        for activity in day.get("activities", [])
                    )
                    if failed_assignment:
                        self._assignment_nogoods.setdefault(macro_key, set()).add(failed_assignment)
                    # Preserve the violation class for targeted recovery.
                    kind = "budget_infeasible" if any("budget_exceeded" in error for error in verification["errors"]) else "verification_failed"
                    result = SolveResult(
                        SEARCH_EXHAUSTED,
                        "verification failed",
                        conflict={"kind": kind, "verification_errors": verification["errors"]},
                    )
                    last_result = result

                requests: List[AcquisitionRequest] = []
                recovery_progress = False
                include_followups = False
                if result.status == NEEDS_INFO:
                    requests = result.missing
                elif result.status in {INFEASIBLE, COVERAGE_GAP, SEARCH_EXHAUSTED}:
                    requests = self._expansion_requests(result.conflict)
                    include_followups = bool(requests)
                elif result.status == PROVEN_INFEASIBLE:
                    if (result.certificate is not None
                            and self._verify_infeasibility_certificate(result.certificate)):
                        break
                    self.metrics["infeasibility_certificates_rejected"] += 1
                    result = SolveResult(
                        COVERAGE_GAP, "proven-infeasible status lacked a valid certificate",
                        conflict={"kind": "invalid_infeasibility_certificate"},
                    )
                    last_result = result
                made_progress = False
                if requests:
                    made_progress = self._run_acquisition_queue(
                        requests, include_followups=include_followups,
                    )
                if made_progress:
                    continue

                incumbent_summary = None
                if best_incumbent is not None:
                    incumbent_summary = {
                        "score": best_incumbent["score"],
                        "verification_errors": best_incumbent["verification"].get("errors") or [],
                        "evidence_coverage": best_incumbent["verification"].get("evidence_coverage"),
                        "budget_total": best_incumbent["verification"].get("budget_total"),
                    }
                action, reflection_requests = self._reflect_on_search(
                    user_query, result, incumbent_summary,
                )
                recovery_progress = action == "REPAIRED"
                if reflection_requests:
                    recovery_progress = self._run_acquisition_queue(
                        reflection_requests, include_followups=True,
                    ) or recovery_progress
                if not recovery_progress:
                    break

            # Exact search + deterministic retrieval + bounded reflection have all
            # stalled.  Hand control back to the solver for minimum-violation
            # feasibility restoration rather than asking the LLM to weaken the IR.
            authoritative_ir = copy.deepcopy(self.problem_ir)
            restored = self._attempt_feasibility_restoration(
                authoritative_ir, last_result, best_incumbent,
            )

            failure = {
                "reason": "structured_solver_failed",
                "last_status": last_result.status if last_result else None,
                "last_reason": last_result.reason if last_result else None,
                "conflict": last_result.conflict if last_result else None,
                "certificate": (
                    last_result.certificate.__dict__
                    if last_result and last_result.certificate else None
                ),
            }

            if restored is not None:
                verification = restored["verification"]
                report = restored["report"]
                self.relaxation_report = report
                self.state.solution = restored["solution"]
                self.state.verification = verification
                self.state.converted_plan = self.to_converted_plan(restored["solution"])
                self.metrics["evidence_coverage"] = verification.get("evidence_coverage")
                self.metrics["relaxation_hard_violations"] = report.get("hard_violation_count", 0)
                plan = self._render_plan(self.state.converted_plan)
                self.metrics["elapsed_time"] = time.time() - started

                if verification.get("success"):
                    # Relaxed search may discover an exact plan that the original
                    # search order missed; it is still an exact success under the
                    # authoritative IR.
                    self.metrics["verification_success"] = True
                    self.metrics["relaxation_exact_recovered"] = True
                    self.metrics["reflection_recovery_success"] = (
                        self.metrics["reflection_actions_accepted"] > 0
                    )
                    return plan, self._diagnostic_payload()

                self.metrics["best_effort_returned"] = True
                self.metrics["relaxed_feasible_returned"] = True
                failure["relaxation"] = report
                return plan, self._diagnostic_payload(extra=failure)

            # Preserve the pre-existing anytime behavior if restoration cannot build
            # a structurally valid candidate. Re-verify the incumbent against the
            # final authoritative IR in case reflection repaired semantics after it
            # was first recorded.
            if best_incumbent is not None:
                incumbent_verification = self._verify_against_problem_ir(
                    best_incumbent["solution"], authoritative_ir,
                )
                self.state.solution = best_incumbent["solution"]
                self.state.verification = incumbent_verification
                self.state.converted_plan = self.to_converted_plan(best_incumbent["solution"])
                self.metrics["best_effort_returned"] = True
                self.metrics["evidence_coverage"] = incumbent_verification.get("evidence_coverage")
                plan = self._render_plan(self.state.converted_plan)
                self.metrics["elapsed_time"] = time.time() - started
                failure["best_incumbent_score"] = best_incumbent["score"]
                return plan, self._diagnostic_payload(extra=failure)
            self.metrics["elapsed_time"] = time.time() - started
            return "", self._diagnostic_payload(extra=failure)

        except Exception as e:
            failure = {"reason": "agent_exception", "error": str(e)}
            self.metrics["elapsed_time"] = time.time() - started
            return "", self._diagnostic_payload(extra=failure)

    def _diagnostic_payload(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "problem_ir": self.problem_ir,
            "metrics": self.metrics,
            "semantic_events": self.semantic_events,
            "verification": self.state.verification if self.state else None,
            "converted_plan": self.state.converted_plan if self.state else None,
            "state_summary": self.state.compact_summary() if self.state else None,
            "solution": self.state.solution if self.state else None,
            "relaxation": self.relaxation_report,
            "constraint_summary": ({
                "truth": self.last_propagation.truth,
                "satisfied": self.last_propagation.satisfied,
                "unresolved": self.last_propagation.unresolved,
                "conflicts": self.last_propagation.conflicts,
                "missing_fact_keys": [request.key for request in self.last_propagation.missing],
            } if self.last_propagation else None),
        }
        if extra:
            payload["failure"] = extra
        return payload


# =============================================================================
# 8. Batch runner
# =============================================================================


def run_agent_inference(
    model: str,
    language: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    output_dir: Path,
    workers: int = 1,
    max_llm_calls: int = 100,
    max_tool_calls: int = 400,
    max_solver_rounds: int = 24,
    max_reflections: int = 2,
    max_relaxation_rounds: int = 12,
    max_relaxed_candidates: int = 8,
    rerun_ids: Optional[List[int]] = None,
    enable_semantic_audit: bool = True,
    enable_reflection: bool = True,
    enable_relaxation: bool = True,
) -> Dict[str, Any]:
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if rerun_ids is not None:
        wanted = {str(x) for x in rerun_ids}
        test_data = [x for x in test_data if str(x.get("id")) in wanted]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "converted_plans").mkdir(exist_ok=True)
    lock = Lock()
    results: List[Dict[str, Any]] = []

    def process(sample: Dict[str, Any]) -> Dict[str, Any]:
        raw_id = sample.get("id", "unknown")
        sid = f"id_{raw_id}" if str(raw_id).isdigit() else str(raw_id)
        query = sample.get("query", "")
        trajectory_path = output_dir / "trajectories" / f"{sid}.json"
        report_path = output_dir / "reports" / f"{sid}.txt"
        converted_path = output_dir / "converted_plans" / f"{sid}_converted.json"
        # Prevent a failed rerun from being evaluated against a stale artifact.
        for stale_path in (trajectory_path, report_path, converted_path):
            stale_path.unlink(missing_ok=True)
        with lock:
            print(f"🚀 {sid}")
        try:
            agent = SolverInformedTravelAgent(
                model=model,
                sample_id=raw_id,
                database_base_path=database_dir,
                tool_schema_path=tool_schema_path,
                language=language,
                max_llm_calls=max_llm_calls,
                max_tool_calls=max_tool_calls,
                max_solver_rounds=max_solver_rounds,
                max_reflections=max_reflections,
                max_relaxation_rounds=max_relaxation_rounds,
                max_relaxed_candidates=max_relaxed_candidates,
                enable_semantic_audit=enable_semantic_audit,
                enable_reflection=enable_reflection,
                enable_relaxation=enable_relaxation,
            )
            plan, diagnostic = agent.run(query)
            internal_verified = bool((diagnostic.get("verification") or {}).get("success"))
            result = {
                "id": sid,
                "query": query,
                "model": model,
                "language": language,
                "final_plan": plan,
                "produced_plan": bool(plan),
                "structured_ready": bool(diagnostic.get("solution")),
                "internal_verified": internal_verified,
                "relaxed_feasible": bool(
                    (diagnostic.get("relaxation") or {}).get("status") == RELAXED_FEASIBLE
                ),
                "official_case_acc": None,
                "success": internal_verified,
                **diagnostic,
            }
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            if plan:
                report_path.write_text(plan, encoding="utf-8")
            if plan and diagnostic.get("converted_plan"):
                with open(converted_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostic["converted_plan"], f, ensure_ascii=False, indent=2)
            with lock:
                failure_info = result.get("failure") or {}
                status = failure_info.get("last_status") or (failure_info.get("reason") if not result["success"] else None)
                suffix = f" | status={status}" if status else ""
                print(
                    f"✅ {sid} | success={result['success']} | "
                    f"tools={result['metrics']['tool_calls']} | llm={result['metrics']['llm_calls']}{suffix}"
                )
            return result
        except Exception as e:
            with lock:
                print(f"❌ {sid}: {e}")
            return {
                "id": sid, "query": query, "produced_plan": False,
                "structured_ready": False, "internal_verified": False,
                "relaxed_feasible": False,
                "official_case_acc": None,
                "success": False, "error": str(e),
                "failure": {"reason": "runner_exception", "error": str(e)},
            }

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(process, sample) for sample in test_data]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: str(r.get("id")))
    success = sum(bool(r.get("success")) for r in results)
    reflected = sum((r.get("metrics") or {}).get("reflection_calls", 0) > 0 for r in results)
    reflection_recovered = sum(
        bool((r.get("metrics") or {}).get("reflection_recovery_success")) for r in results
    )
    failure_status_counts: Dict[str, int] = {}
    failure_reason_counts: Dict[str, int] = {}
    query_outcome_counts: Dict[str, int] = {}
    for result in results:
        failure = result.get("failure") or {}
        if not result.get("success"):
            status = str(failure.get("last_status") or failure.get("reason") or "UNKNOWN_FAILURE")
            reason = str(failure.get("last_reason") or failure.get("reason") or "UNKNOWN_FAILURE")
            failure_status_counts[status] = failure_status_counts.get(status, 0) + 1
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
        state_summary = result.get("state_summary") or {}
        for outcome in (state_summary.get("query_outcomes") or {}).values():
            status = str((outcome or {}).get("status") or "UNKNOWN")
            query_outcome_counts[status] = query_outcome_counts.get(status, 0) + 1
    summary = {
        "total": len(results),
        "success": success,
        "internal_verified": success,
        "produced_plans": sum(bool(r.get("produced_plan")) for r in results),
        "best_effort_plans": sum(
            bool((r.get("metrics") or {}).get("best_effort_returned")) for r in results
        ),
        "relaxed_plans": sum(bool(r.get("relaxed_feasible")) for r in results),
        "relaxation_attempted_cases": sum(
            bool((r.get("metrics") or {}).get("relaxation_attempted")) for r in results
        ),
        "relaxation_exact_recovered": sum(
            bool((r.get("metrics") or {}).get("relaxation_exact_recovered")) for r in results
        ),
        "reflected_cases": reflected,
        "reflection_recovered": reflection_recovered,
        "reflection_recovery_rate": reflection_recovered / reflected if reflected else 0.0,
        "failed": len(results) - success,
        "failure_status_counts": failure_status_counts,
        "failure_reason_counts": failure_reason_counts,
        "query_outcome_counts": query_outcome_counts,
        "avg_llm_calls": (sum((r.get("metrics") or {}).get("llm_calls", 0) for r in results) / len(results)) if results else 0,
        "avg_tool_calls": (sum((r.get("metrics") or {}).get("tool_calls", 0) for r in results) / len(results)) if results else 0,
        "avg_reflection_calls": (sum((r.get("metrics") or {}).get("reflection_calls", 0) for r in results) / len(results)) if results else 0,
        "results": results,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return summary


# =============================================================================
# 9. CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Solver-informed DeepPlanning Travel agent")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--max-tool-calls", type=int, default=400)
    parser.add_argument("--max-solver-rounds", type=int, default=24)
    parser.add_argument("--max-reflections", type=int, default=2)
    parser.add_argument("--max-relaxation-rounds", type=int, default=12)
    parser.add_argument("--max-relaxed-candidates", type=int, default=8)
    parser.add_argument("--ids", nargs="*", type=int, default=None)
    parser.add_argument("--no-semantic-audit", action="store_true",
                        help="disable the independent NL-to-IR semantic audit")
    parser.add_argument("--no-reflection", "--no-semantic-recovery", "--no-constraint-repair",
                        dest="no_reflection", action="store_true",
                        help="disable the unified conflict-recovery/reflection controller")
    parser.add_argument("--no-relaxation", action="store_true",
                        help="disable solver-based minimum-violation feasibility restoration")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--database-dir", type=str, default=None)
    parser.add_argument("--tool-schema", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", help="run official evaluator on the selected cases after inference")
    args = parser.parse_args()

    anchor = Path(__file__).resolve()
    domain_root = anchor.parent.parent
    test_data = Path(args.test_data) if args.test_data else domain_root / "data" / f"travelplanning_query_{args.language}.json"
    data_root = Path(os.environ.get("BENCHMARK_DATA_DIR", str(domain_root.parent / "data")))
    database_dir = Path(args.database_dir) if args.database_dir else data_root / f"database_{args.language}"
    tool_schema = Path(args.tool_schema) if args.tool_schema else domain_root / "tools" / f"tool_schema_{args.language}.json"
    output_dir = Path(args.output_dir) if args.output_dir else domain_root / "results" / f"solver_informed_{args.model}_{args.language}"

    summary = run_agent_inference(
        model=args.model,
        language=args.language,
        test_data_path=test_data,
        database_dir=database_dir,
        tool_schema_path=tool_schema,
        output_dir=output_dir,
        workers=args.workers,
        max_llm_calls=args.max_llm_calls,
        max_tool_calls=args.max_tool_calls,
        max_solver_rounds=args.max_solver_rounds,
        max_reflections=args.max_reflections,
        max_relaxation_rounds=args.max_relaxation_rounds,
        max_relaxed_candidates=args.max_relaxed_candidates,
        rerun_ids=args.ids,
        enable_semantic_audit=not args.no_semantic_audit,
        enable_reflection=not args.no_reflection,
        enable_relaxation=not args.no_relaxation,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    if args.evaluate:
        sys.path.insert(0, str(domain_root.parent))
        from travelplanning.evaluation.eval_converted import evaluate_plans

        with open(test_data, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if args.ids is not None:
            wanted = {str(i) for i in args.ids}
            samples = [sample for sample in samples if str(sample.get("id")) in wanted]
        selected_data = output_dir / "selected_test_data.json"
        selected_data.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        evaluation = evaluate_plans(
            result_dir=output_dir,
            test_data_path=selected_data,
            database_dir=database_dir,
            workers=args.workers,
        )
        metrics = evaluation.get("metrics") or {}
        plans_evaluated = int(evaluation.get("total") or 0)
        total_cases = len(samples)
        print(json.dumps({
            "total_cases": total_cases,
            "plans_evaluated": plans_evaluated,
            "delivery_rate": plans_evaluated / total_cases if total_cases else 0.0,
            "case_acc": metrics.get("case_acc", 0.0),
            "composite_score": metrics.get("composite_score", 0.0),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
