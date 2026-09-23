"""Prompts used by the structured shopping planner and its ReAct fallback."""


PARSER_PROMPT = """You compile a shopping request into strict JSON. Return JSON only.

Schema:
{{
  "slots": [
    {{
      "slot_id": 1,
      "quantity": 1,
      "constraints": [
        {{"field": "name", "op": "contains", "value": "running shoe", "source_text": "running shoes"}}
      ]
    }}
  ],
  "global_constraints": {{}}
}}

Create one slot for every distinct requested product. Extract only constraints explicitly stated
in the request and preserve strict inequalities. Do not apply user-profile defaults and do not
copy global requirements into individual slots; Python normalization performs those operations.
Represent a total-price interval once in global_constraints as
{{"budget": {{"min": 1000.0, "max": 1500.0}}}}. Do not turn a total budget into a per-product
price constraint. Do not invent product types or name constraints from non-literal paraphrases.
slot_id values must be consecutive integers starting at 1.

Allowed fields: {allowed_fields}
Allowed operators: {allowed_ops}
"""


FALLBACK_REACT_PROMPT_LEVEL_1 = """You are repairing a shopping cart after a structured planner
reported a specific error. Use the supplied repair context and tools to satisfy every product slot.
Choose the lowest-price feasible product for each slot, remove extra items, verify the final cart,
and finish with one JSON object describing the actual cart."""


FALLBACK_REACT_PROMPT_LEVEL_2 = """You are repairing a shopping cart after a structured planner
reported a specific error. Use the supplied repair context and tools to satisfy every product slot
and both bounds of the total-price budget. Remove extra items, verify the final cart and budget,
and finish with one JSON object describing the actual cart."""


FALLBACK_REACT_PROMPT_LEVEL_3 = """You are repairing a shopping cart after a structured planner
reported a specific error. Use the supplied repair context and tools to satisfy every product slot
and reproduce a valid product-and-coupon plan. Respect coupon ownership, VIP eligibility, combined
threshold usage, and quantities. Verify the final cart and finish with one JSON object."""


# Compatibility names used by shoppingplanning/run.py. The structured path only uses these when
# deterministic verification requests a ReAct fallback.
SYSTEM_PROMPT_level1 = FALLBACK_REACT_PROMPT_LEVEL_1
SYSTEM_PROMPT_level2 = FALLBACK_REACT_PROMPT_LEVEL_2
SYSTEM_PROMPT_level3 = FALLBACK_REACT_PROMPT_LEVEL_3


class PromptLib:
    """Namespace retained for the existing runner interface."""


prompt_lib = PromptLib()
prompt_lib.PARSER_PROMPT = PARSER_PROMPT
prompt_lib.SYSTEM_PROMPT_level1 = SYSTEM_PROMPT_level1
prompt_lib.SYSTEM_PROMPT_level2 = SYSTEM_PROMPT_level2
prompt_lib.SYSTEM_PROMPT_level3 = SYSTEM_PROMPT_level3
