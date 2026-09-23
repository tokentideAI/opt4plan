"""Exhaustive filters for structured textual product attributes."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from base_shopping_tool import BaseShoppingTool, register_tool


def _tokens(value: Any) -> set[str]:
    text = re.sub(r"[’']s\b", "", str(value).casefold())
    aliases = {
        "female": "women", "woman": "women", "womens": "women",
        "male": "men", "man": "men", "mens": "men",
        "shoes": "shoe", "sneakers": "sneaker", "boots": "boot",
        "loafers": "loafer", "sandals": "sandal", "slippers": "slipper",
        "pants": "pant", "trousers": "trouser", "shorts": "short",
        "joggers": "jogger", "skirts": "skirt", "leggings": "legging",
        "jeans": "jean", "shirts": "shirt", "tops": "top", "tees": "tee",
        "jackets": "jacket", "coats": "coat", "hoodies": "hoodie",
        "sweaters": "sweater", "seasons": "season",
    }
    ignored = {"a", "an", "and", "the"}
    return {
        aliases.get(token, token)
        for token in re.findall(r"[a-z0-9]+", text)
        if token not in ignored
    }


def _canonical(value: Any) -> str:
    tokens = _tokens(value)
    return " ".join(sorted(tokens))


@register_tool("filter_by_text")
class FilterByTextTool(BaseShoppingTool):
    """Filter the complete catalog (or an ID subset) without top-k truncation."""

    ALLOWED_FIELDS = {
        "name", "brand", "color", "size",
        "suitable_season", "target_demographic",
    }

    def __init__(self, cfg: Dict = None):
        super().__init__(cfg)
        self.products: List[Dict[str, Any]] = []
        self.products_map: Dict[str, Dict[str, Any]] = {}
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "database", "case_0", "products.jsonl"
        )
        path = (
            Path(self.cfg["database_path"]) / "products.jsonl"
            if self.cfg and self.cfg.get("database_path")
            else Path(default_path)
        )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        product = json.loads(line)
                        self.products.append(product)
                        self.products_map[product["product_id"]] = product
        except FileNotFoundError:
            pass

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            args = self._verify_json_format_args(params)
        except ValueError as error:
            return self.format_result_as_json({"error": str(error)})

        field = args.get("condition_key")
        operator = args.get("operator")
        value = args.get("value")
        product_ids = args.get("product_ids")
        if field not in self.ALLOWED_FIELDS:
            return self.format_result_as_json({"error": f"unsupported text field: {field}"})
        if operator not in {"==", "contains"} or value is None:
            return self.format_result_as_json({"error": "operator must be == or contains"})

        if product_ids is not None:
            missing = [pid for pid in product_ids if pid not in self.products_map]
            if missing:
                return self.format_result_as_json({
                    "error": f"Some product_ids not found in database: {missing}"
                })
            search_space = [self.products_map[pid] for pid in product_ids]
        else:
            search_space = self.products

        expected_tokens = _tokens(value)
        expected_canonical = _canonical(value)
        matches = []
        for product in search_space:
            actual = product.get(field)
            if actual is None:
                continue
            matched = (
                _canonical(actual) == expected_canonical
                if operator == "=="
                else bool(expected_tokens) and expected_tokens.issubset(_tokens(actual))
            )
            if matched:
                matches.append(product["product_id"])

        return self.format_result_as_json({
            "filtered_products_ids": matches,
            "input_count": len(search_space),
            "total_count": len(matches),
            "returned_count": len(matches),
            "is_exhaustive": True,
        })
