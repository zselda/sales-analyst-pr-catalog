"""
ING Product Catalog — real product taxonomy for grounded suggestions
=====================================================================
Loads data/product_catalog.json (extracted from the bank's
product_categories.numbers file) and renders a prompt section that
grounds product_analyst's cross-sell suggestions in REAL ING products.

Catalog shape: ANA ÜRÜN (main product) → ALT ÜRÜN (sub-product), each
sub-product tagged with currency (TL / YP=foreign currency) and optional
AÇIKLAMA conditions.

Suggestion policy enforced via the prompt:
1. Suggest at ANA ÜRÜN level first; drill into ALT ÜRÜN only when the
   Mizan signals give enough information (currency, import vs export, ...).
2. YP (foreign-currency) products only when FX / foreign-trade evidence
   exists in the signals.
3. Respect AÇIKLAMA conditions (EXIMBANK / Reeskont only-if-current;
   PARA TRANSFERLERİ is an indirect hook, not a direct product).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("swarm.agents.product_catalog")

CATALOG_PATH = Path(__file__).parent.parent / "data" / "product_catalog.json"

# Signals that prove foreign-currency / cross-border activity → gate YP products
_FX_SIGNAL_KEYS = {
    "FX Net Impact (646/656)",
    "Export Revenue (601)",
    "Trade Finance Signals (159/340)",
    "SWIFT/Transfer Expenses",
}

_catalog_cache = None


def load_product_catalog(path: Path = CATALOG_PATH) -> dict:
    """Load (and cache) the ING product catalog JSON."""
    global _catalog_cache
    if _catalog_cache is None:
        try:
            with open(path, encoding="utf-8") as f:
                _catalog_cache = json.load(f)
        except Exception as e:
            logger.error(f"Could not load product catalog: {e}")
            _catalog_cache = {"metadata": {}, "conditions": {}, "categories": []}
    return _catalog_cache


def _active_signal_keys(product_signals: dict) -> set:
    """Signal keys with real (non-zero) volume or balance evidence."""
    active = set()
    for key, data in (product_signals or {}).items():
        if not isinstance(data, dict):
            continue
        if abs(float(data.get("volume") or 0)) > 0 or abs(float(data.get("balance") or 0)) > 0:
            active.add(key)
    return active


def build_catalog_injection(product_signals: dict) -> dict:
    """
    Build the product-catalog prompt addition for product_analyst.

    Returns {"system_addition": str, "user_addition": str}.
    Only ANA ÜRÜN categories whose trigger signals are active are listed,
    so the model is grounded in products that the data actually supports.
    """
    catalog = load_product_catalog()
    categories = catalog.get("categories", [])
    conditions = catalog.get("conditions", {})
    if not categories:
        return {"system_addition": "", "user_addition": ""}

    active = _active_signal_keys(product_signals)
    has_fx = bool(active & _FX_SIGNAL_KEYS)

    active_cats, inactive_cats = [], []
    for cat in categories:
        triggers = set(cat.get("trigger_signal_keys", []))
        (active_cats if (triggers & active) else inactive_cats).append(cat)

    used_conditions = set()

    def _render_cat(cat: dict) -> str:
        lines = [f"### {cat['ana_urun']} ({cat.get('en', '')})"]
        if cat.get("category_condition"):
            used_conditions.add(cat["category_condition"])
            lines.append(f"  ⚠️ CONDITION [{cat['category_condition']}]")
        if cat.get("requires_fx") and not has_fx:
            lines.append("  (foreign-currency category — only if FX/foreign-trade evidence appears)")
        alt_parts = []
        for alt in cat.get("alt_urunler", []):
            tag = alt.get("currency", "")
            extra = []
            if alt.get("trade_direction"):
                extra.append(alt["trade_direction"])
            if alt.get("condition"):
                used_conditions.add(alt["condition"])
                extra.append(f"COND:{alt['condition']}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            alt_parts.append(f"{alt['name']} [{tag}]{suffix}")
        lines.append("  ALT ÜRÜN: " + "; ".join(alt_parts))
        return "\n".join(lines)

    parts = []
    if active_cats:
        parts.append(
            "## 🗂️ ING PRODUCT CATALOG — REAL PRODUCT REFERENCE (not an exhaustive limit)\n"
            "ANA ÜRÜN categories below are ACTIVE for this company (their "
            "trigger signals fired). PREFER these real product names when a "
            "match exists — but you are NOT limited to this list: signal-driven "
            "and few-shot opportunities beyond the catalog are still valid "
            "(see the few-shot examples and the policy in the system prompt).\n\n"
            + "\n".join(_render_cat(c) for c in active_cats)
        )
    if inactive_cats:
        parts.append(
            "## 🗂️ CROSS-SELL CATALOG (no direct signal — sector/greenfield only)\n"
            + "; ".join(c["ana_urun"] for c in inactive_cats)
        )
    if used_conditions:
        cond_lines = "\n".join(
            f"- [{key}] {conditions.get(key, '')}" for key in sorted(used_conditions)
        )
        parts.append("## ⚠️ AÇIKLAMA CONDITIONS (must respect)\n" + cond_lines)

    fx_state = "PRESENT" if has_fx else "ABSENT"
    user_addition = "\n\n" + "\n\n".join(parts)

    system_addition = (
        "\n\n## PRODUCT-CATALOG SUGGESTION POLICY (REAL ING PRODUCTS)\n"
        "Use the ING PRODUCT CATALOG in the user message as the PRIMARY "
        "reference for real product naming (ANA ÜRÜN → ALT ÜRÜN). It grounds "
        "your wording — it is NOT an exhaustive ceiling on what you may "
        "recommend. Rules:\n"
        "- DEPTH: Recommend at ANA ÜRÜN (main product) level FIRST. Drill into "
        "a specific ALT ÜRÜN only when the Mizan signals give enough information "
        "to justify it (e.g. clear currency, import vs export, check vs loan). "
        "If unsure, stay at ANA ÜRÜN level.\n"
        f"- CURRENCY: YP means foreign currency (Yabancı Para). Foreign-currency "
        f"evidence in this company is {fx_state}. Suggest YP [YP]-tagged "
        f"ALT ÜRÜN only when FX / export / import / foreign-transfer evidence "
        f"exists; otherwise prefer TL [TL] products.\n"
        "- CONDITIONS: Strictly honor any AÇIKLAMA condition. EXIMBANK and TCMB "
        "Reeskont products may be recommended ONLY with concrete evidence of a "
        "current such loan. PARA TRANSFERLERİ is not a directly sellable product "
        "— use high transfer volume only as a cash-flow capture hook.\n"
        "- BEYOND THE CATALOG (IMPORTANT): When a product signal or a few-shot "
        "scenario reveals a genuine opportunity whose product is NOT in the "
        "catalog, you SHOULD still recommend it — e.g. a Payroll package from "
        "heavy personnel spending (720/730/760/770), Leasing from a machinery "
        "park (253/301/401), or Factoring from a notes-receivable portfolio "
        "(121). Use cross-sector few-shot patterns freely; prefer a catalog "
        "name when one matches, otherwise name the standard ING product.\n"
        "- ANTI-HALLUCINATION: Never fabricate sub-account codes or invent "
        "products that do not exist at ING. Every recommendation must be backed "
        "by a cited signal/account or an established few-shot pattern."
    )
    logger.info(
        f"🗂️ Product catalog injection: {len(active_cats)} active ANA ÜRÜN "
        f"categories, FX evidence {fx_state}, {len(used_conditions)} conditions"
    )
    return {"system_addition": system_addition, "user_addition": user_addition}
