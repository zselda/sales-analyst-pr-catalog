"""
Agent 2: Quantitative Analyst — Local calc + LLM interpretation

Refactored to use BaseAgent for tracing and error isolation.

DYNAMIC MAPPING: Account code descriptions (hesap_kodu_aciklama) are
extracted dynamically from each Mizan document — no hardcoded dictionaries.

HIERARCHICAL AGGREGATION: Balances roll up from leaf nodes to parents
using prefix-based tree structure. Both period movement (borc/alacak)
and closing balance (borc_bakiye/alacak_bakiye) are computed.
"""

import logging
import pandas as pd
from collections import defaultdict
from agents.base import BaseAgent
from llm_config import invoke_llm, QUANT_ANALYST_SYSTEM_PROMPT

logger = logging.getLogger("swarm.agents.quant_analyst")


# ── Dynamic Mapping & Hierarchy Functions ──────────────────────────


def _build_dynamic_mapping(df: pd.DataFrame) -> dict:
    """
    Build hesap_kodu -> hesap_kodu_aciklama mapping dynamically
    from the actual Mizan document data.

    This replaces ANY hardcoded/static dictionary. Every document
    will produce its own mapping reflecting exactly what it contains.
    """
    mapping = {}
    for _, row in df.iterrows():
        code = str(row.get("account_code", "")).strip()
        name = str(row.get("account_name", "")).strip()
        if code and name and name != "nan":
            mapping[code] = name
    logger.info(f"Built dynamic mapping: {len(mapping)} account codes")
    return mapping


def _build_hierarchy_tree(df: pd.DataFrame) -> dict:
    """
    Build a hierarchical tree from account codes using '.' delimiter.

    Returns a dict keyed by root codes, each containing:
    {
        "code": "101",
        "name": "ALINAN ÇEKLER",
        "children": {
            "101.010": {
                "code": "101.010",
                "name": "ALINAN ÇEKLER",
                "children": {
                    "101.010.001": {..., "children": {}, "is_leaf": True},
                    "101.010.002": {..., "children": {}, "is_leaf": True},
                }
            }
        }
    }
    """
    tree = {}
    all_codes = sorted(df["account_code"].astype(str).unique())

    for code in all_codes:
        parts = code.split(".")
        name_row = df[df["account_code"] == code]
        name = str(name_row.iloc[0].get("account_name", code)) if not name_row.empty else code

        # Navigate/create path in tree
        current_level = tree
        accumulated = ""
        for i, part in enumerate(parts):
            accumulated = part if i == 0 else f"{accumulated}.{part}"
            if accumulated not in current_level:
                current_level[accumulated] = {
                    "code": accumulated,
                    "name": name if accumulated == code else "",
                    "children": {},
                    "is_leaf": False,
                }
            if accumulated == code:
                current_level[accumulated]["name"] = name
            current_level = current_level[accumulated]["children"]

    # Mark leaf nodes (no children)
    def _mark_leaves(node_dict):
        for key, node in node_dict.items():
            if not node["children"]:
                node["is_leaf"] = True
            else:
                _mark_leaves(node["children"])

    _mark_leaves(tree)
    return tree


def _get_leaf_codes(tree_node: dict) -> list:
    """Recursively collect all leaf codes under a tree node."""
    leaves = []
    children = tree_node.get("children", {})
    if not children:
        return [tree_node["code"]]
    for child in children.values():
        leaves.extend(_get_leaf_codes(child))
    return leaves


def _aggregate_hierarchy(df: pd.DataFrame, code: str, tree: dict,
                         col_borc="debit", col_alacak="credit",
                         col_borc_bakiye="balance_debit",
                         col_alacak_bakiye="balance_credit") -> dict:
    """
    Compute aggregated balances for a code using hierarchical leaf-node summation.

    Returns:
        {
            "period_borc": float,     # sum of borc (debit) from leaves
            "period_alacak": float,   # sum of alacak (credit) from leaves
            "period_movement": float, # borc - alacak
            "closing_borc_bakiye": float,
            "closing_alacak_bakiye": float,
            "closing_balance": float, # borc_bakiye - alacak_bakiye
            "leaf_count": int,
            "leaf_codes": list,
            "reported_value": float | None,  # from parent row if exists
            "validation_status": str,  # "match", "mismatch", "no_parent_row"
        }
    """
    # Find this code's node in the tree
    node = _find_node_in_tree(code, tree)

    if node is None:
        # Code not in tree — try direct DataFrame lookup
        m = df[df["account_code"] == code]
        if not m.empty:
            row = m.iloc[0]
            borc = float(row.get(col_borc, 0))
            alacak = float(row.get(col_alacak, 0))
            b_bak = float(row.get(col_borc_bakiye, 0))
            a_bak = float(row.get(col_alacak_bakiye, 0))
            return {
                "period_borc": borc, "period_alacak": alacak,
                "period_movement": borc - alacak,
                "closing_borc_bakiye": b_bak, "closing_alacak_bakiye": a_bak,
                "closing_balance": b_bak - a_bak,
                "leaf_count": 1, "leaf_codes": [code],
                "reported_value": None, "validation_status": "leaf_direct",
            }
        return {
            "period_borc": 0, "period_alacak": 0, "period_movement": 0,
            "closing_borc_bakiye": 0, "closing_alacak_bakiye": 0,
            "closing_balance": 0,
            "leaf_count": 0, "leaf_codes": [],
            "reported_value": None, "validation_status": "not_found",
        }

    # Collect leaf codes
    leaf_codes = _get_leaf_codes(node)

    if not leaf_codes:
        leaf_codes = [code]

    # Sum from leaf nodes
    leaf_df = df[df["account_code"].isin(leaf_codes)]
    period_borc = float(leaf_df[col_borc].sum()) if col_borc in leaf_df.columns else 0
    period_alacak = float(leaf_df[col_alacak].sum()) if col_alacak in leaf_df.columns else 0
    closing_b = float(leaf_df[col_borc_bakiye].sum()) if col_borc_bakiye in leaf_df.columns else 0
    closing_a = float(leaf_df[col_alacak_bakiye].sum()) if col_alacak_bakiye in leaf_df.columns else 0

    # Cross-validate against parent row if it exists
    parent_row = df[df["account_code"] == code]
    reported_value = None
    validation_status = "no_parent_row"

    if not parent_row.empty and len(leaf_codes) > 1:
        # Parent row exists and has children — validate
        rep_borc = float(parent_row.iloc[0].get(col_borc, 0))
        rep_alacak = float(parent_row.iloc[0].get(col_alacak, 0))
        reported_value = rep_borc - rep_alacak
        computed_value = period_borc - period_alacak

        if abs(reported_value - computed_value) <= 0.01:
            validation_status = "match"
        else:
            validation_status = "mismatch"
            logger.warning(
                f"⚠️ Hierarchy mismatch for {code}: "
                f"reported={reported_value:,.2f}, computed={computed_value:,.2f}, "
                f"diff={abs(reported_value - computed_value):,.2f}"
            )
    elif len(leaf_codes) == 1 and leaf_codes[0] == code:
        validation_status = "leaf_direct"

    return {
        "period_borc": period_borc,
        "period_alacak": period_alacak,
        "period_movement": period_borc - period_alacak,
        "closing_borc_bakiye": closing_b,
        "closing_alacak_bakiye": closing_a,
        "closing_balance": closing_b - closing_a,
        "leaf_count": len(leaf_codes),
        "leaf_codes": leaf_codes,
        "reported_value": reported_value,
        "validation_status": validation_status,
    }


def _find_node_in_tree(code: str, tree: dict):
    """Find a node in the hierarchy tree by its code."""
    if code in tree:
        return tree[code]
    for key, node in tree.items():
        found = _find_node_in_tree(code, node.get("children", {}))
        if found is not None:
            return found
    return None


class QuantAnalystAgent(BaseAgent):
    name = "quant_analyst"
    description = "Calculate financial ratios from standardized Mizan data with dynamic mapping"
    required_inputs = ["standardized_mizan"]
    output_keys = ["financial_ratios"]

    def execute(self, state: dict) -> dict:
        retry_count = state.get("retry_count", 0)
        verification_errors = state.get("verification_errors", "")

        logger.info(f"Calculating ratios (attempt #{retry_count + 1})...")
        if verification_errors:
            logger.info(f"Verification feedback: {verification_errors}")

        # ── Read Dönem (time period) from pipeline state ──
        donem_info = state.get("donem_info") or {"period_days": 360, "period_months": 12, "raw": "unknown", "label": "Annual (12M) — default"}
        period_days = donem_info.get("period_days", 360)
        period_months = donem_info.get("period_months", 12)
        donem_label = donem_info.get("label", "Unknown")
        logger.info(f"Using Dönem: {donem_label} ({period_days} days)")

        standardized = state.get("standardized_mizan", [])
        if not standardized:
            return {"financial_ratios": {"error": "No standardized mizan data"}, "retry_count": retry_count + 1}

        df = pd.DataFrame(standardized)

        # ── STEP 1: Build dynamic mapping from document ──
        aciklama_map = _build_dynamic_mapping(df)

        # ── STEP 2: Build hierarchy tree ──
        hierarchy_tree = _build_hierarchy_tree(df)
        logger.info(f"Hierarchy tree built: {len(hierarchy_tree)} root nodes")

        # ── STEP 3: Define balance functions ──

        def bal(code: str) -> float:
            """Get net balance for an account code (exact match or prefix sum). UNCHANGED."""
            # Try exact match first
            m = df[df["account_code"] == code]
            if not m.empty:
                return float(m.iloc[0]["debit"] - m.iloc[0]["credit"])
            # Fall back to prefix matching (sum all sub-accounts)
            m = df[df["account_code"].str.startswith(code)]
            if not m.empty:
                return float(m["debit"].sum() - m["credit"].sum())
            return 0.0

        def bal_named(code: str) -> dict:
            """
            Hierarchy-aware balance with dynamic mapping, dual balances,
            and data validation (double-check).

            Returns enriched dict with:
            - value: absolute closing balance (primary metric for ratios)
            - hesap_kodu / hesap_kodu_aciklama: from dynamic mapping
            - period_movement: borc - alacak (period activity)
            - closing_balance: borc_bakiye - alacak_bakiye (remaining)
            - children_sum: computed from leaf aggregation
            - validation_status: match/mismatch/leaf_direct/not_found
            """
            agg = _aggregate_hierarchy(df, code, hierarchy_tree)

            # Determine description from dynamic mapping
            desc = aciklama_map.get(code, "")
            if not desc:
                # Try to find closest parent with a name
                prefix = code.split(".")[0]
                desc = aciklama_map.get(prefix, f"HESAP {code}")

            # Primary value: use closing balance when available, fallback to period movement
            closing = agg["closing_balance"]
            period = agg["period_movement"]
            primary_value = closing if closing != 0 else period

            return {
                "value": abs(primary_value),
                "hesap_kodu": code,
                "hesap_kodu_aciklama": desc,
                "period_movement": period,
                "period_borc": agg["period_borc"],
                "period_alacak": agg["period_alacak"],
                "closing_balance": closing,
                "closing_borc_bakiye": agg["closing_borc_bakiye"],
                "closing_alacak_bakiye": agg["closing_alacak_bakiye"],
                "leaf_count": agg["leaf_count"],
                "leaf_codes": agg["leaf_codes"],
                "children_sum": abs(period),
                "validation_status": agg["validation_status"],
            }

        # ── LOCAL CALCULATIONS ──
        # Profitability Metrics
        revenue_600 = abs(bal("600"))
        cogs_620 = abs(bal("620"))
        gross_profit = revenue_600 - cogs_620
        gross_margin = ((gross_profit) / revenue_600 * 100) if revenue_600 else 0

        op_exp_630 = abs(bal("630"))
        op_exp_631 = abs(bal("631"))
        op_exp_632 = abs(bal("632"))
        op_expenses = op_exp_630 + op_exp_631 + op_exp_632
        operating_profit = gross_profit - op_expenses
        operating_margin = (operating_profit / revenue_600 * 100) if revenue_600 else 0

        # Liquidity Metrics
        ca_codes = df[df["category"] == "Current Assets"]["account_code"].tolist()
        current_assets = float(df[df["category"] == "Current Assets"]["debit"].sum())
        stl_codes = df[df["category"] == "Short-Term Liabilities"]["account_code"].tolist()
        short_term_liab = float(df[df["category"] == "Short-Term Liabilities"]["credit"].sum())

        current_ratio = (current_assets / short_term_liab) if short_term_liab else 0

        inv_150 = abs(bal("150"))
        inv_151 = abs(bal("151"))
        inv_152 = abs(bal("152"))
        inv_153 = abs(bal("153"))
        inventory = inv_150 + inv_151 + inv_152 + inv_153
        quick_ratio = ((current_assets - inventory) / short_term_liab) if short_term_liab else 0

        # Checks & Liquid Instruments
        received_checks_101 = abs(bal("101"))
        given_checks_103 = abs(bal("103"))

        # ── COMPETITOR BANK ANALYSIS (102, 300, 400) ──
        def get_bank_breakdown(parent_code: str) -> list:
            """Extracts sub-account balances to identify competitor bank shares."""
            shares = []
            total_bal = 0
            for _, row in df.iterrows():
                code = str(row["account_code"])
                # Identify sub-accounts (e.g., '102.01', '102 01')
                if code.startswith(parent_code) and code != parent_code:
                    net_bal = abs(float(row["debit"]) - float(row["credit"]))
                    if net_bal > 0:
                        # Use dynamic mapping for name
                        name = aciklama_map.get(code, str(row.get("account_name", code)))
                        shares.append({
                            "name": name,
                            "balance": net_bal
                        })
                        total_bal += net_bal

            # Calculate percentages
            for s in shares:
                s["share_pct"] = (s["balance"] / total_bal * 100) if total_bal else 0

            # Sort highest balance first
            shares.sort(key=lambda x: x["balance"], reverse=True)
            return shares

        banks_102 = get_bank_breakdown("102")
        banks_300 = get_bank_breakdown("300")
        banks_400 = get_bank_breakdown("400")

        def fmt_shares(shares):
            if not shares:
                return "No detailed sub-account data available."
            return ", ".join([f"{s['name']} (₺{s['balance']:,.0f} - %{s['share_pct']:.1f})" for s in shares[:5]])

        # Leverage & Debt Metrics
        total_liab = float(
            df[df["category"] == "Short-Term Liabilities"]["credit"].sum() +
            df[df["category"] == "Long-Term Liabilities"]["credit"].sum()
        )
        total_equity = float(df[df["category"] == "Equity"]["credit"].sum())
        debt_to_equity = (total_liab / total_equity) if total_equity else 0

        short_term_loans_300 = abs(bal("300"))
        credit_card_expenses_309 = abs(bal("309"))
        long_term_loans_400 = abs(bal("400"))
        total_bank_loans = short_term_loans_300 + credit_card_expenses_309 + long_term_loans_400
        bank_debt_ratio = (total_bank_loans / total_liab * 100) if total_liab else 0

        fin_exp_780 = abs(bal("780"))
        fin_expense_ratio = (fin_exp_780 / revenue_600 * 100) if revenue_600 else 0

        # Efficiency & Working Capital
        trade_recv_120 = abs(bal("120"))
        trade_recv_121 = abs(bal("121"))
        trade_receivables = trade_recv_120 + trade_recv_121
        collection_period = (trade_receivables / revenue_600 * period_days) if revenue_600 else 0

        trade_pay_320 = abs(bal("320"))
        trade_pay_321 = abs(bal("321"))
        trade_payables = trade_pay_320 + trade_pay_321
        payment_period = (trade_payables / cogs_620 * period_days) if cogs_620 else 0

        # Transactional / Behavioral Metrics
        pos_780_01 = abs(bal("780.01"))
        pos_ratio = (pos_780_01 / revenue_600 * 100) if revenue_600 else 0

        # ── HIERARCHY ENRICHMENT: bal_named() with validation ──
        # Log validation results for key accounts
        key_accounts = ["100", "101", "102", "120", "150", "300", "320", "400", "500", "600", "620", "780"]
        validation_warnings = []
        for kc in key_accounts:
            named = bal_named(kc)
            if named["validation_status"] == "mismatch":
                validation_warnings.append(
                    f"{kc}-{named['hesap_kodu_aciklama']}: "
                    f"leaf_sum={named['children_sum']:,.0f} vs reported"
                )

        if validation_warnings:
            logger.warning(f"🔍 Data validation warnings:\n" + "\n".join(f"  ⚠️ {w}" for w in validation_warnings))

        # ── Build hierarchy summary for key accounts ──
        def hierarchy_detail(code: str) -> dict:
            """Get hierarchy detail including children breakdown for a code."""
            named = bal_named(code)
            node = _find_node_in_tree(code, hierarchy_tree)
            children_detail = []
            if node and node.get("children"):
                for child_code, child_node in node["children"].items():
                    child_named = bal_named(child_code)
                    leaf_detail = []
                    if child_node.get("children"):
                        for leaf_code, leaf_node in child_node["children"].items():
                            leaf_named = bal_named(leaf_code)
                            leaf_detail.append({
                                "code": leaf_code,
                                "name": leaf_named["hesap_kodu_aciklama"],
                                "period_movement": leaf_named["period_movement"],
                                "closing_balance": leaf_named["closing_balance"],
                            })
                    children_detail.append({
                        "code": child_code,
                        "name": child_named["hesap_kodu_aciklama"],
                        "period_movement": child_named["period_movement"],
                        "closing_balance": child_named["closing_balance"],
                        "leaves": leaf_detail,
                    })
            return {
                **named,
                "children_detail": children_detail,
            }

        account_hierarchy = {
            code: hierarchy_detail(code) for code in ["101", "102", "120", "150", "300", "320", "400"]
        }

        ratios = {
            "gross_margin": {
                "value": round(gross_margin, 2), "unit": "%",
                "formula": "(Revenue[600] - COGS[620]) / Revenue[600] × 100",
                "accounts_used": ["600", "620"],
                "raw_values": {
                    "revenue_600": bal_named("600"),
                    "cogs_620": bal_named("620"),
                    "gross_profit": gross_profit,
                }
            },
            "operating_margin": {
                "value": round(operating_margin, 2), "unit": "%",
                "formula": "(Gross Profit - Operating Expenses[630+631+632]) / Revenue[600] × 100",
                "accounts_used": ["600", "620", "630", "631", "632"],
                "raw_values": {
                    "revenue_600": bal_named("600"),
                    "cogs_620": bal_named("620"),
                    "op_exp_630": bal_named("630"),
                    "op_exp_631": bal_named("631"),
                    "op_exp_632": bal_named("632"),
                    "operating_profit": operating_profit,
                }
            },
            "current_ratio": {
                "value": round(current_ratio, 2), "unit": "x",
                "formula": "Current Assets [1xx] / Short-Term Liabilities [3xx]",
                "accounts_used": ca_codes,
                "raw_values": {
                    "current_assets": current_assets,
                    "short_term_liabilities": short_term_liab,
                }
            },
            "quick_ratio": {
                "value": round(quick_ratio, 2), "unit": "x",
                "formula": "(Current Assets[1xx] - Inventory[15x]) / Short-Term Liabilities[3xx]",
                "accounts_used": ca_codes + ["150", "151", "152", "153"],
                "raw_values": {
                    "liquid_assets": current_assets - inventory,
                    "short_term_liabilities": short_term_liab,
                    "received_checks_101": bal_named("101"),
                    "given_checks_103": bal_named("103"),
                    "inventory_150": bal_named("150"),
                    "inventory_151": bal_named("151"),
                    "inventory_152": bal_named("152"),
                    "inventory_153": bal_named("153"),
                }
            },
            "collection_period": {
                "value": round(collection_period, 0), "unit": "days",
                "formula": f"Trade Receivables[120+121] / Revenue[600] × {period_days}",
                "accounts_used": ["120", "121", "600"],
                "period_days_used": period_days,
                "raw_values": {
                    "trade_receivables_120": bal_named("120"),
                    "trade_receivables_121": bal_named("121"),
                    "trade_receivables": trade_receivables,
                    "revenue_600": bal_named("600"),
                }
            },
            "payment_period": {
                "value": round(payment_period, 0), "unit": "days",
                "formula": f"Trade Payables[320+321] / COGS[620] × {period_days}",
                "accounts_used": ["320", "321", "620"],
                "period_days_used": period_days,
                "raw_values": {
                    "trade_payables_320": bal_named("320"),
                    "trade_payables_321": bal_named("321"),
                    "trade_payables": trade_payables,
                    "cogs_620": bal_named("620"),
                }
            },
            "debt_to_equity": {
                "value": round(debt_to_equity, 2), "unit": "x",
                "formula": "Total Liabilities [3xx+4xx] / Equity [5xx]",
                "accounts_used": stl_codes + ["400", "500", "570"],
                "raw_values": {
                    "total_liabilities": total_liab,
                    "total_equity": total_equity,
                    "sermaye_500": bal_named("500"),
                    "gecmis_yil_karlari_570": bal_named("570"),
                }
            },
            "bank_debt_ratio": {
                "value": round(bank_debt_ratio, 2), "unit": "%",
                "formula": "Bank Loans[300+309+400] / Total Liabilities × 100",
                "accounts_used": ["300", "309", "400"],
                "raw_values": {
                    "banka_kredileri_kv_300": bal_named("300"),
                    "diger_mali_borclar_309": bal_named("309"),
                    "banka_kredileri_uv_400": bal_named("400"),
                    "total_bank_loans": total_bank_loans,
                    "total_liabilities": total_liab,
                }
            },
            "financial_expense_ratio": {
                "value": round(fin_expense_ratio, 2), "unit": "%",
                "formula": "Financial Expenses [780] / Revenue [600] × 100",
                "accounts_used": ["780", "600"],
                "raw_values": {
                    "finansman_giderleri_780": bal_named("780"),
                    "revenue_600": bal_named("600"),
                }
            },
            "pos_commission_ratio": {
                "value": round(pos_ratio, 2), "unit": "%",
                "formula": "POS Commission [780.01] / Revenue [600] × 100",
                "accounts_used": ["780.01", "600"],
                "raw_values": {
                    "pos_komisyon_780_01": bal_named("780.01"),
                    "revenue_600": bal_named("600"),
                }
            },
            # Competitor data for downstream Strategist
            "competitor_banks": {
                "102": banks_102,
                "300": banks_300,
                "400": banks_400
            },
            # Hierarchical account tree for downstream agents
            "account_hierarchy": account_hierarchy,
            # Dynamic mapping for reference
            "dynamic_mapping": aciklama_map,
            # Data validation results
            "validation_warnings": validation_warnings,
            # Temporal context
            "donem_context": {
                "raw": donem_info.get("raw"),
                "year": donem_info.get("year"),
                "period_months": period_months,
                "period_days": period_days,
                "label": donem_label,
                "annualization_factor": round(12 / period_months, 2) if period_months else 1.0,
            },
        }

        for n, d in ratios.items():
            if isinstance(d, dict) and "value" in d:
                logger.info(f"  - {n}: {d['value']}{d['unit']}")

        # ── LLM INTERPRETATION ──
        llm_text = ""
        try:
            summary = "\n".join(
                f"- {n}: {d['value']}{d['unit']} (accounts: {d['accounts_used']})"
                for n, d in ratios.items() if isinstance(d, dict) and "value" in d
            )

            # Build hierarchy breakdown strings for key accounts
            def fmt_hierarchy(code: str) -> str:
                h = account_hierarchy.get(code, {})
                children = h.get("children_detail", [])
                if not children:
                    return f"  {code}-{h.get('hesap_kodu_aciklama', '?')}: " \
                           f"Period={h.get('period_movement', 0):,.0f}, " \
                           f"Closing={h.get('closing_balance', 0):,.0f}"
                lines = [f"  {code}-{h.get('hesap_kodu_aciklama', '?')} (total):"]
                for child in children:
                    lines.append(
                        f"    └─ {child['code']}-{child['name']}: "
                        f"Period={child['period_movement']:,.0f}, "
                        f"Closing={child['closing_balance']:,.0f}"
                    )
                    for leaf in child.get("leaves", []):
                        lines.append(
                            f"       └─ {leaf['code']}-{leaf['name']}: "
                            f"Period={leaf['period_movement']:,.0f}, "
                            f"Closing={leaf['closing_balance']:,.0f}"
                        )
                return "\n".join(lines)

            # Get enriched bal_named data for key accounts
            bn_600 = bal_named("600")
            bn_620 = bal_named("620")
            bn_101 = bal_named("101")
            bn_103 = bal_named("103")

            hierarchy_section = "\n".join(
                fmt_hierarchy(c) for c in ["101", "102", "120", "300", "400"]
            )

            validation_section = ""
            if validation_warnings:
                validation_section = (
                    "\n### ⚠️ DATA VALIDATION WARNINGS:\n" +
                    "\n".join(f"- {w}" for w in validation_warnings)
                )

            prompt = (
                f"⏱️ DATA PERIOD: {donem_label} ({period_days} days). "
                f"All turnover/period metrics are scaled to this timeframe.\n\n"
                f"Analyze these financial ratios for {state.get('company_name', 'Company')} ({state.get('sector', 'General')}):\n\n"
                f"{summary}\n\n"
                f"### RAW DATA (Dynamically Extracted — Hesap Kodu Açıklama from Document):\n"
                f"- **600-{bn_600['hesap_kodu_aciklama']}:** Period=₺{bn_600['period_movement']:,.0f}, Closing=₺{bn_600['closing_balance']:,.0f}\n"
                f"- **620-{bn_620['hesap_kodu_aciklama']}:** Period=₺{abs(bal('620')):,.0f}\n"
                f"- **630-{bal_named('630')['hesap_kodu_aciklama']}:** ₺{op_exp_630:,.0f} | **631-{bal_named('631')['hesap_kodu_aciklama']}:** ₺{op_exp_631:,.0f} | **632-{bal_named('632')['hesap_kodu_aciklama']}:** ₺{op_exp_632:,.0f}\n"
                f"- **Dönen Varlıklar (1xx):** ₺{current_assets:,.0f} | **150-153 STOKLAR:** ₺{inventory:,.0f} | **Kısa Vadeli Borçlar (3xx):** ₺{short_term_liab:,.0f}\n"
                f"- **120-{bal_named('120')['hesap_kodu_aciklama']}:** ₺{trade_recv_120:,.0f} | **121-{bal_named('121')['hesap_kodu_aciklama']}:** ₺{trade_recv_121:,.0f}\n"
                f"- **320-{bal_named('320')['hesap_kodu_aciklama']}:** ₺{trade_pay_320:,.0f} | **321-{bal_named('321')['hesap_kodu_aciklama']}:** ₺{trade_pay_321:,.0f}\n"
                f"- **101-{bn_101['hesap_kodu_aciklama']}:** Period=₺{bn_101['period_movement']:,.0f}, Closing=₺{bn_101['closing_balance']:,.0f}\n"
                f"- **103-{bn_103['hesap_kodu_aciklama']}:** Period=₺{abs(bal('103')):,.0f}\n"
                f"- **Total Liabilities:** ₺{total_liab:,.0f} | **300-{bal_named('300')['hesap_kodu_aciklama']}:** ₺{short_term_loans_300:,.0f} | **309-{bal_named('309')['hesap_kodu_aciklama']}:** ₺{credit_card_expenses_309:,.0f} | **400-{bal_named('400')['hesap_kodu_aciklama']}:** ₺{long_term_loans_400:,.0f}\n"
                f"- **500-{bal_named('500')['hesap_kodu_aciklama']}:** ₺{abs(bal('500')):,.0f} | **570-{bal_named('570')['hesap_kodu_aciklama']}:** ₺{abs(bal('570')):,.0f} | **Total Equity:** ₺{total_equity:,.0f}\n"
                f"- **780-{bal_named('780')['hesap_kodu_aciklama']}:** ₺{fin_exp_780:,.0f} | **780.01-{bal_named('780.01')['hesap_kodu_aciklama']}:** ₺{pos_780_01:,.0f}\n\n"
                f"### HIERARCHICAL ACCOUNT BREAKDOWNS:\n{hierarchy_section}\n\n"
                f"### COMPETITOR BANK DISTRIBUTION:\n"
                f"- **102-BANKALAR (Deposits):** {fmt_shares(banks_102)}\n"
                f"- **300-BANKA KREDİLERİ KV (ST Loans):** {fmt_shares(banks_300)}\n"
                f"- **400-BANKA KREDİLERİ UV (LT Loans):** {fmt_shares(banks_400)}\n\n"
                f"{validation_section}\n\n"
                f"Based on your system instructions, structure your analysis into these four exact pillars:\n"
                f"1. PROFITABILITY\n"
                f"2. LIQUIDITY & WORKING CAPITAL\n"
                f"3. LEVERAGE & DEPENDENCY\n"
                f"4. TRANSACTIONAL COST\n\n"
                f"IMPORTANT: Cite every value with its hesap kodu and dynamically-extracted açıklama.\n"
                f"For each key account, analyze BOTH the period movement (Borç/Alacak) AND the closing balance (Bakiye) to assess financial dynamics.\n"
                f"Identify red flags and map the mathematical groundwork for downstream cross-selling, explicitly utilizing the competitor bank distribution.\n\n"
                f"Note to Strategist: These financial metrics reflect a {donem_label} snapshot ({period_days} days). "
                f"Annualized projections should multiply period values by {12/period_months:.1f}x."
            )
            llm_text = invoke_llm(QUANT_ANALYST_SYSTEM_PROMPT, prompt, temperature=0.2, max_tokens=1500)
            self.metrics.record_llm_call(tokens=len(llm_text.split()))
            logger.info(f"✅ LLM interpretation: {len(llm_text)} chars")
        except Exception as e:
            logger.warning(f"LLM skipped: {e}")
            llm_text = "LLM interpretation unavailable."

        ratios["llm_interpretation"] = llm_text
        return {"financial_ratios": ratios, "retry_count": retry_count + 1}


# Module-level callable for LangGraph
quant_analyst_agent = QuantAnalystAgent()