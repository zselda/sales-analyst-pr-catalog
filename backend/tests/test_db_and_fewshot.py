"""
Tests — Local DB enrichment, selective few-shot injection, sector analysis
===========================================================================
No LLM required: all LLM calls are mocked.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.few_shot_library import (
    FEW_SHOT_SCENARIOS,
    classify_product_opportunities,
    build_few_shot_injection,
)


# ============================================================================
# Few-shot library / product classification
# ============================================================================

ACTIVE_SIGNALS = {
    "POS Collection (108)": {"volume": 85_000_000, "balance": 2_000_000},
    "Received Checks (101)": {"volume": 40_000_000, "balance": 5_000_000},
    "Bank Loans (300 + 400)": {"volume": 90_000_000, "balance": 35_000_000},
    "Total Financial Expenses (780)": {"volume": 9_000_000, "balance": 0},
    "Fleet Assets (254)": {"volume": 0, "balance": 0},  # inactive
}


class TestFewShotClassification:
    def test_only_active_signals_selected(self):
        cls = classify_product_opportunities(ACTIVE_SIGNALS, sector="Trading")
        assert "checks" in cls["selected_keys"]
        assert "pos" in cls["selected_keys"]
        assert "cash_loan" in cls["selected_keys"]
        # No data evidence → never selected
        assert "insurance" not in cls["selected_keys"]
        assert "insurance" in cls["inactive_keys"]

    def test_used_products_are_excluded(self):
        cls = classify_product_opportunities(
            ACTIVE_SIGNALS, sector="Trading", product_flags={"pos": 1}
        )
        assert "pos" not in cls["selected_keys"]
        assert [e["product_key"] for e in cls["excluded_existing"]] == ["pos"]

    def test_max_scenarios_cap(self):
        signals = {}
        for sc in FEW_SHOT_SCENARIOS:
            for k in sc["signal_keys"]:
                signals[k] = {"volume": 1_000_000, "balance": 500_000}
        cls = classify_product_opportunities(signals, max_scenarios=6)
        assert len(cls["selected"]) == 6

    def test_injection_blocks(self):
        cls = classify_product_opportunities(
            ACTIVE_SIGNALS, sector="Trading", product_flags={"pos": 1}
        )
        inj = build_few_shot_injection(cls, sector="Trading")
        assert "MIZAN ANALYSIS CORE RULES" in inj["system_addition"]
        assert "DO NOT RECOMMEND" in inj["user_addition"]
        assert "Scenario [checks]" in inj["user_addition"]
        assert "Scenario [pos]" not in inj["user_addition"]

    def test_empty_signals_no_injection(self):
        cls = classify_product_opportunities({}, sector="General")
        inj = build_few_shot_injection(cls)
        assert inj["system_addition"] == ""
        assert inj["user_addition"] == ""

    def test_sector_priority_aligned_with_tcmb_benchmarks(self):
        """Every TCMB benchmark sector must have a SECTOR_PRODUCT_PRIORITY entry."""
        import json
        from pathlib import Path
        from agents.few_shot_library import SECTOR_PRODUCT_PRIORITY, _sector_matches
        bench_path = Path(__file__).parent.parent / "data" / "tcmb_sector_benchmarks.json"
        tcmb_sectors = json.loads(bench_path.read_text(encoding="utf-8"))["sectors"].keys()
        priority_keys = list(SECTOR_PRODUCT_PRIORITY.keys())
        for sector in tcmb_sectors:
            assert any(_sector_matches(sector, [k]) for k in priority_keys), (
                f"TCMB sector '{sector}' has no SECTOR_PRODUCT_PRIORITY entry"
            )

    def test_multi_sector_priority_lines_and_fallback(self):
        from agents.few_shot_library import build_few_shot_injection
        cls = classify_product_opportunities(ACTIVE_SIGNALS, sector="Retail + Export")
        inj = build_few_shot_injection(cls, sector="Retail + Export")
        # Both the Retail and Export priority entries must be injected
        assert "SECTOR PRIORITY (Retail / Perakende)" in inj["system_addition"]
        assert "SECTOR PRIORITY (Export / İhracat)" in inj["system_addition"]
        # Unmatched sector → General fallback line
        inj_fb = build_few_shot_injection(cls, sector="Quantum Basket Weaving")
        assert "General fallback" in inj_fb["system_addition"]

    def test_multi_sector_affinity_boost(self):
        """'Retail + Export' must trigger the Retail affinity boost in scoring."""
        from agents.few_shot_library import _sector_matches
        assert _sector_matches("Retail + Export", ["Retail", "Trading"]) is True
        assert _sector_matches("Textile Manufacturing + Export", ["Üretim", "Manufacturing"]) is True
        assert _sector_matches("Tourism", ["Retail", "Trading"]) is False

    def test_scenario_signal_keys_match_product_analyst_outputs(self):
        """
        DYNAMIC completeness check: run the real ProductAnalystAgent on mock
        Mizan data (LLM mocked) and assert that EVERY few-shot scenario's
        signal_keys exist in the produced product_signals dict. Catches any
        scenario added without its extraction signal (and vice-versa drift).
        """
        import builtins
        from unittest.mock import patch
        from mock_data import get_mizan_df
        from agents.data_ingestion import data_ingestion_agent

        state = {
            "tax_id": "x", "company_name": "T", "sector": "General",
            "mizan_data": get_mizan_df().to_dict(orient="records"),
            "retry_count": 0,
            "agent_metrics": {}, "execution_timeline": [], "error_log": [],
        }
        state.update(data_ingestion_agent(state))
        real_print = builtins.print
        builtins.print = lambda *a, **k: None  # silence the agent's prompt dump
        try:
            with patch("agents.product_analyst.invoke_llm", return_value="MOCK"):
                from agents.product_analyst import product_analyst_agent
                out = product_analyst_agent(state)
        finally:
            builtins.print = real_print

        assert out["agent_metrics"]["product_analyst"]["status"] == "success"
        produced = set(out["product_signals"].keys())
        missing = {
            sc["product_key"]: [k for k in sc["signal_keys"] if k not in produced]
            for sc in FEW_SHOT_SCENARIOS
            if any(k not in produced for k in sc["signal_keys"])
        }
        assert not missing, f"Scenario signal keys missing from product_analyst: {missing}"


# ============================================================================
# Local DB layer + enrichment agent
# ============================================================================

class TestLocalDB:
    @pytest.fixture
    def tmp_db(self, tmp_path):
        return tmp_path / "test_customer_db.sqlite"

    def test_seed_and_snapshot(self, tmp_db):
        from local_db import seed_demo_data, get_customer_snapshot
        seed_demo_data(tmp_db)
        snap = get_customer_snapshot("1234567890", db_path=tmp_db)
        assert snap["found"] is True
        assert snap["matched_by"] == "tax_id"
        # Long-format metrics keyed by ENGLISH description in the snapshot
        assert snap["financial_metrics"]["Acid-Test Ratio (Quick Ratio)"] == 0.95
        assert snap["financial_metrics"]["Current Ratio"] == 1.40
        # Whitespace-tolerant mapping: 'Nakit  Döngüsü (Gün)' (double space)
        # and 'Toplam Net Borç / VAFÖK ' (trailing space) must map cleanly
        assert snap["financial_metrics"]["Cash Conversion Cycle (Days)"] == 74
        assert snap["financial_metrics"]["Total Net Debt / EBITDA"] == 2.1
        assert snap["product_flags"]["pos"] == 1
        assert snap["product_flags"]["credit_card"] == 0
        # LONG DataFrame with both TR and EN description columns
        df = snap["financials_df"]
        assert not df.empty
        assert {"tr_description", "value", "en_description"} <= set(df.columns)

    def test_company_name_fallback(self, tmp_db):
        from local_db import seed_demo_data, get_customer_snapshot
        seed_demo_data(tmp_db)
        snap = get_customer_snapshot("unknown-tax-id", company_name="Mizan Best",
                                     db_path=tmp_db)
        assert snap["found"] is True
        assert snap["matched_by"] == "company_name"

    def test_missing_customer_graceful(self, tmp_db):
        from local_db import seed_demo_data, get_customer_snapshot
        seed_demo_data(tmp_db)
        snap = get_customer_snapshot("0000000000", company_name="Nope", db_path=tmp_db)
        assert snap["found"] is False
        assert snap["financial_metrics"] == {}
        assert snap["product_flags"] == {}

    def test_unmapped_tr_description_falls_back_to_turkish(self):
        import pandas as pd
        from local_db import add_english_descriptions
        df = pd.DataFrame({
            "tr_description": ["Cari Oran", "Bilinmeyen Yeni Metrik"],
            "value": [1.4, 9.9],
        })
        out = add_english_descriptions(df)
        assert out.loc[0, "en_description"] == "Current Ratio"
        assert out.loc[1, "en_description"] == "Bilinmeyen Yeni Metrik"  # TR kept

    def test_legacy_wide_schema_migrated_to_long(self, tmp_db):
        import sqlite3
        from local_db import get_connection
        # Simulate a pre-existing DB with the old WIDE schema
        conn = sqlite3.connect(str(tmp_db))
        conn.execute(
            "CREATE TABLE customer_financials ("
            "tax_id TEXT, company_name TEXT, period TEXT, "
            "acid_test_ratio REAL, gross_profit REAL)"
        )
        conn.commit()
        conn.close()
        conn = get_connection(tmp_db)  # migration fires here
        cols = {row[1] for row in conn.execute("PRAGMA table_info(customer_financials)")}
        conn.close()
        assert "tr_description" in cols and "value" in cols
        assert "acid_test_ratio" not in cols

    def test_nace_code_roundtrip(self, tmp_db):
        from local_db import seed_demo_data, get_customer_snapshot
        seed_demo_data(tmp_db)
        snap = get_customer_snapshot("1234567890", db_path=tmp_db)
        assert snap["nace_code"] == "47.11.02"

    def test_db_enrichment_agent_outputs(self, monkeypatch, tmp_db):
        import local_db
        from agents.db_enrichment import db_enrichment_agent
        local_db.seed_demo_data(tmp_db)
        monkeypatch.setattr("agents.db_enrichment.ensure_db", lambda: None)
        monkeypatch.setattr(
            "agents.db_enrichment.get_customer_snapshot",
            lambda tax_id, company_name: local_db.get_customer_snapshot(
                tax_id, company_name, db_path=tmp_db),
        )
        out = db_enrichment_agent({"tax_id": "1234567890", "company_name": "X"})
        assert out["db_meta"]["found"] is True
        assert out["db_product_flags"]["pos"] == 1
        assert "Acid-Test Ratio (Quick Ratio)" in out["db_financial_metrics"]


# ============================================================================
# Oracle EDW backend (mocked — no real warehouse needed)
# ============================================================================

class TestOracleBackend:
    @pytest.fixture
    def oracle_env(self, monkeypatch):
        monkeypatch.setenv("CUSTOMER_DB_BACKEND", "oracle")

    def _long_fin_df(self):
        import pandas as pd
        return pd.DataFrame({
            "tax_id": ["999"] * 2, "company_name": ["EDW CO"] * 2,
            "period": ["202606"] * 2,
            "tr_description": ["Asit Test Oranı", "Cari Oran"],
            "value": [0.9, 1.3],
        })

    def test_backend_selection_via_env(self, oracle_env):
        from local_db import get_backend
        assert get_backend() == "oracle"

    def test_read_oracle_lowercases_columns(self, oracle_env, monkeypatch):
        """Oracle returns UPPERCASE identifiers; _read_oracle must normalize."""
        import pandas as pd
        import local_db
        from unittest.mock import MagicMock
        fake_engine = MagicMock()
        fake_engine.connect.return_value.__enter__ = MagicMock(return_value="conn")
        fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(local_db, "_get_oracle_engine", lambda: fake_engine)
        upper_df = pd.DataFrame({"TR_DESCRIPTION": ["Cari Oran"], "VALUE": [1.3]})
        monkeypatch.setattr(pd, "read_sql", lambda *a, **k: upper_df)
        out = local_db._read_oracle("SELECT 1")
        assert list(out.columns) == ["tr_description", "value"]

    def test_financials_df_routed_through_oracle(self, oracle_env, monkeypatch):
        import local_db
        captured = {}
        def fake_read(query, params=None):
            captured["query"], captured["params"] = query, params
            return self._long_fin_df()
        monkeypatch.setattr(local_db, "_read_oracle", fake_read)
        df = local_db.get_financials_df("999")
        assert "CUSTOMER_FINANCIALS" in captured["query"]
        assert captured["params"] == {"tax_id": "999"}
        assert "en_description" in df.columns          # TR→EN preprocessing applied
        assert "Current Ratio" in set(df["en_description"])

    def test_oracle_failure_returns_empty_no_demo_fallback(self, oracle_env, monkeypatch):
        import local_db
        def boom(query, params=None):
            raise RuntimeError("EDW unreachable")
        monkeypatch.setattr(local_db, "_read_oracle", boom)
        assert local_db.get_financials_df("999").empty
        assert local_db.get_product_flags_df("999").empty
        snap = local_db.get_customer_snapshot("999", company_name="EDW CO")
        assert snap["found"] is False
        assert snap["financial_metrics"] == {}

    def test_snapshot_from_oracle(self, oracle_env, monkeypatch):
        import pandas as pd
        import local_db
        prod_df = pd.DataFrame([{"tax_id": "999", "company_name": "EDW CO",
                                 "nace_code": "47.11.01", "pos": 1, "checks": 0}])
        def fake_read(query, params=None):
            return self._long_fin_df() if "FINANCIALS" in query else prod_df
        monkeypatch.setattr(local_db, "_read_oracle", fake_read)
        snap = local_db.get_customer_snapshot("999")
        assert snap["found"] is True and snap["matched_by"] == "tax_id"
        assert snap["financial_metrics"]["Acid-Test Ratio (Quick Ratio)"] == 0.9
        assert snap["product_flags"]["pos"] == 1
        assert snap["nace_code"] == "47.11.01"

    def test_oracle_backend_is_read_only(self, oracle_env):
        from local_db import upsert_financials, upsert_product_flags, seed_demo_data
        with pytest.raises(NotImplementedError):
            upsert_financials("x", "202606", {"Cari Oran": 1.0})
        with pytest.raises(NotImplementedError):
            upsert_product_flags("x", {"pos": 1})
        with pytest.raises(NotImplementedError):
            seed_demo_data()

    def test_ensure_db_noop_on_oracle(self, oracle_env, tmp_path):
        from local_db import ensure_db
        target = tmp_path / "should_not_exist.sqlite"
        ensure_db(db_path=target)
        assert not target.exists()   # no sqlite bootstrap on EDW backend


# ============================================================================
# Product Recommendation Catalog (deterministic matrix membership)
# ============================================================================

class TestRecommendationCatalog:
    def _many_signals(self):
        """Signals that activate MORE scenarios than the 6-scenario prompt cap."""
        return {
            "POS Collection (108)": {"volume": 85_000_000, "balance": 2_000_000},
            "Received Checks (101)": {"volume": 40_000_000, "balance": 5_000_000},
            "Bank Loans (300 + 400)": {"volume": 90_000_000, "balance": 35_000_000},
            "Total Financial Expenses (780)": {"volume": 9_000_000, "balance": 0},
            "Payroll & Personnel": {"volume": 15_000_000, "balance": 0},
            "FX Net Impact (646/656)": {"volume": 20_000_000, "balance": 0},
            "Notes Receivable (121)": {"volume": 35_000_000, "balance": 12_000_000},
            "Bank Transaction Volume (102)": {"volume": 250_000_000, "balance": 8_000_000},
            "Machinery & Equipment (253)": {"volume": 12_000_000, "balance": 45_000_000},
        }

    def test_includes_all_active_signals_beyond_prompt_cap(self):
        from agents.few_shot_library import build_recommendation_catalog
        catalog = build_recommendation_catalog(self._many_signals(), sector="Manufacturing")
        active_keys = {e["product_key"] for e in catalog["active"]}
        # 9+ scenarios are active — the catalog must NOT cap at 6
        assert len(active_keys) > 6
        for expected in ["pos", "checks", "cash_loan", "payroll", "fx",
                         "factoring", "cash_management", "deposit", "leasing"]:
            assert expected in active_keys, f"{expected} missing from catalog"
        assert catalog["total_rows"] == len(catalog["active"]) + len(catalog["cross_sell"])

    def test_deterministic_across_runs(self):
        from agents.few_shot_library import build_recommendation_catalog
        a = build_recommendation_catalog(self._many_signals(), sector="Manufacturing",
                                         product_flags={"pos": 1})
        b = build_recommendation_catalog(self._many_signals(), sector="Manufacturing",
                                         product_flags={"pos": 1})
        assert a == b  # identical content AND order

    def test_used_products_never_in_catalog(self):
        from agents.few_shot_library import build_recommendation_catalog
        catalog = build_recommendation_catalog(
            self._many_signals(), sector="Manufacturing",
            product_flags={"pos": 1, "payroll": 1},
        )
        all_keys = {e["product_key"]
                    for e in catalog["active"] + catalog["cross_sell"]}
        assert "pos" not in all_keys and "payroll" not in all_keys
        excluded = {e["product_key"] for e in catalog["excluded_existing"]}
        assert excluded == {"pos", "payroll"}

    def test_cross_sell_sector_matched_capped_and_tagged(self):
        from agents.few_shot_library import build_recommendation_catalog
        # Only one active signal → many inactive scenarios available
        signals = {"POS Collection (108)": {"volume": 1_000_000, "balance": 0}}
        catalog = build_recommendation_catalog(signals, sector="Construction")
        assert len(catalog["cross_sell"]) <= 4
        for e in catalog["cross_sell"]:
            assert e["signal_type"] == "Cross-Sell"
            assert "No current volume" in e["data_evidence"]
        # Construction affinity products should surface as cross-sell
        cross_keys = {e["product_key"] for e in catalog["cross_sell"]}
        assert cross_keys & {"letter_of_guarantee", "leasing", "checks", "insurance", "dbs"}

    def test_evidence_uses_real_signal_values(self):
        from agents.few_shot_library import build_recommendation_catalog
        catalog = build_recommendation_catalog(self._many_signals(), sector="General")
        factoring = next(e for e in catalog["active"] if e["product_key"] == "factoring")
        assert "₺35,000,000" in factoring["data_evidence"]
        assert factoring["signal_type"] == "Active Signal"
        assert "ING Faktoring" in factoring["ing_products"]

    def test_render_catalog_for_prompt(self):
        from agents.few_shot_library import (
            build_recommendation_catalog, render_catalog_for_prompt,
        )
        catalog = build_recommendation_catalog(
            self._many_signals(), sector="Manufacturing", product_flags={"pos": 1}
        )
        rendered = render_catalog_for_prompt(catalog)
        n = catalog["total_rows"]
        assert f"EXACTLY these {n}" in rendered["user_addition"]
        assert f"{n}. [" in rendered["user_addition"]      # last numbered row present
        assert "MATRIX STABILITY RULE" in rendered["system_addition"]
        assert "EXCLUDED" in rendered["user_addition"]     # pos listed as excluded
        # Empty catalog → no additions
        empty = render_catalog_for_prompt({"active": [], "cross_sell": []})
        assert empty["user_addition"] == "" and empty["system_addition"] == ""

    def test_strategist_prompt_contains_catalog(self, monkeypatch):
        from unittest.mock import patch
        import builtins
        state = {
            "tax_id": "x", "company_name": "Test Co", "sector": "Manufacturing",
            "financial_ratios": {"current_ratio": {"value": 1.1}, "donem_context": {}},
            "network_data": {"stats": {}, "nodes": []},
            "product_signals": self._many_signals(),
            "db_product_flags": {},
        }
        cap = {}
        def fake_llm(system_prompt, user_prompt, **kw):
            cap["system"], cap["user"] = system_prompt, user_prompt
            return "MOCK"
        real_print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            with patch("agents.strategist.invoke_llm", side_effect=fake_llm):
                from agents.strategist import sales_strategist_agent
                out = sales_strategist_agent(state)
        finally:
            builtins.print = real_print
        assert out["agent_metrics"]["strategist"]["status"] == "success"
        assert "PRODUCT RECOMMENDATION CATALOG (AUTHORITATIVE" in cap["user"]
        assert "MATRIX STABILITY RULE" in cap["system"]

    def test_strategist_prefers_catalog_from_product_analyst(self):
        from unittest.mock import patch
        import builtins
        marker_catalog = {
            "active": [{
                "product_key": "factoring", "signal_type": "Active Signal",
                "client_need": "MARKER-NEED-12345", "ing_products": ["ING Faktoring"],
                "data_evidence": "• evidence", "reasoning": "r", "signal_strength": 1.0,
            }],
            "cross_sell": [], "excluded_existing": [], "total_rows": 1,
        }
        state = {
            "tax_id": "x", "company_name": "Test Co", "sector": "General",
            "financial_ratios": {"donem_context": {}},
            "network_data": {"stats": {}, "nodes": []},
            "product_signals": {"recommendation_catalog": marker_catalog},
            "db_product_flags": {},
        }
        cap = {}
        def fake_llm(system_prompt, user_prompt, **kw):
            cap["user"] = user_prompt
            return "MOCK"
        real_print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            with patch("agents.strategist.invoke_llm", side_effect=fake_llm):
                from agents.strategist import sales_strategist_agent
                sales_strategist_agent(state)
        finally:
            builtins.print = real_print
        assert "MARKER-NEED-12345" in cap["user"]  # stored catalog used, not rebuilt


# ============================================================================
# ING Product Catalog (ANA ÜRÜN / ALT ÜRÜN grounded suggestions)
# ============================================================================

class TestProductCatalog:
    def test_catalog_integrity(self):
        """13 ANA ÜRÜN, 72 ALT ÜRÜN, conditions resolvable, currencies valid."""
        from agents.product_catalog import load_product_catalog
        cat = load_product_catalog()
        assert len(cat["categories"]) == 13
        assert sum(len(c["alt_urunler"]) for c in cat["categories"]) == 72
        cond_keys = set(cat["conditions"])
        for c in cat["categories"]:
            if c.get("category_condition"):
                assert c["category_condition"] in cond_keys
            for alt in c["alt_urunler"]:
                assert alt["currency"] in ("TL", "YP")
                if alt.get("condition"):
                    assert alt["condition"] in cond_keys

    def test_trigger_keys_are_real_product_signals(self):
        """Every catalog trigger key must be produced by ProductAnalystAgent."""
        import builtins
        from unittest.mock import patch
        from mock_data import get_mizan_df
        from agents.data_ingestion import data_ingestion_agent
        from agents.product_catalog import load_product_catalog

        state = {"tax_id": "x", "company_name": "T", "sector": "General",
                 "mizan_data": get_mizan_df().to_dict(orient="records"),
                 "retry_count": 0, "agent_metrics": {},
                 "execution_timeline": [], "error_log": []}
        state.update(data_ingestion_agent(state))
        real_print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            with patch("agents.product_analyst.invoke_llm", return_value="MOCK"):
                from agents.product_analyst import product_analyst_agent
                out = product_analyst_agent(state)
        finally:
            builtins.print = real_print
        produced = set(out["product_signals"].keys())
        for c in load_product_catalog()["categories"]:
            unknown = [k for k in c["trigger_signal_keys"] if k not in produced]
            assert not unknown, f"{c['ana_urun']} has unknown triggers: {unknown}"

    def test_fx_gating_present(self):
        from agents.product_catalog import build_catalog_injection
        inj = build_catalog_injection({
            "Export Revenue (601)": {"volume": 5e7, "balance": 0},
            "Bank Loans (300 + 400)": {"volume": 9e7, "balance": 3.5e7},
        })
        assert "is PRESENT" in inj["system_addition"]
        assert "DIŞ TİCARET" in inj["user_addition"]        # export → foreign trade active
        # depth rule present
        assert "ANA ÜRÜN (main product) level FIRST" in inj["system_addition"]

    def test_fx_gating_absent(self):
        from agents.product_catalog import build_catalog_injection
        inj = build_catalog_injection({"POS Collection (108)": {"volume": 8e7, "balance": 2e6}})
        assert "is ABSENT" in inj["system_addition"]
        active_block = inj["user_addition"].split("CROSS-SELL")[0]
        assert "ÜYE İŞ YERLERİ VE POS" in active_block       # POS active
        assert "DIŞ TİCARET" not in active_block             # foreign trade NOT active
        assert "DIŞ TİCARET" in inj["user_addition"]         # but listed as cross-sell

    def test_conditions_surface_when_relevant(self):
        from agents.product_catalog import build_catalog_injection
        inj = build_catalog_injection({
            "Bank Loans (300 + 400)": {"volume": 9e7, "balance": 3.5e7},
            "Total Financial Expenses (780)": {"volume": 9e6, "balance": 0},
            "Guarantee Letter Commissions": {"volume": 2e6, "balance": 0},
        })
        # EXIMBANK (TL loans) + Reeskont (non-cash) conditions must appear
        assert "eximbank_current_only" in inj["user_addition"]
        assert "reeskont_current_only" in inj["user_addition"]
        assert "AÇIKLAMA CONDITIONS" in inj["user_addition"]

    def test_empty_signals_no_injection(self):
        from agents.product_catalog import build_catalog_injection
        inj = build_catalog_injection({})
        assert "CROSS-SELL CATALOG" in inj["user_addition"]   # all inactive, listed as cross-sell
        assert "is ABSENT" in inj["system_addition"]

    def test_catalog_is_not_a_hard_limit(self):
        """The catalog grounds naming but must NOT cap recommendations —
        signal/few-shot products beyond it (payroll, leasing, factoring)
        stay allowed, and the old 'do not invent outside catalog' rule is gone."""
        from agents.product_catalog import build_catalog_injection
        inj = build_catalog_injection({"Payroll & Personnel": {"volume": 1.5e7, "balance": 0}})
        sys = inj["system_addition"]
        user = inj["user_addition"]
        assert "Do NOT invent products outside this catalog" not in sys
        assert "not an exhaustive" in user.lower() or "NOT limited to this list" in user
        assert "BEYOND THE CATALOG" in sys
        assert "Payroll package" in sys          # explicit cross-sector example
        # anti-hallucination intent preserved
        assert "ANTI-HALLUCINATION" in sys

    def test_zero_signals_compacted_active_kept(self):
        """Zero-signal breakdown lines collapse to 'no signal' (token saving)
        while active lines keep their full 5-column detail and account map."""
        from unittest.mock import patch
        from agents.product_analyst import ProductAnalystAgent
        rows = [
            {"account_code": "108", "account_name": "Diğer Hazır Değerler",
             "debit": 85_000_000, "credit": 0, "balance_debit": 2_000_000, "balance_credit": 0},
            {"account_code": "600", "account_name": "Yurtiçi Satışlar",
             "debit": 0, "credit": 90_000_000, "balance_debit": 0, "balance_credit": 90_000_000},
        ]
        state = {"tax_id": "x", "company_name": "Sparse Co", "sector": "Retail",
                 "standardized_mizan": rows, "retry_count": 0,
                 "db_product_flags": {}, "db_financial_metrics": {},
                 "agent_metrics": {}, "execution_timeline": [], "error_log": []}
        cap = {}
        with patch("agents.product_analyst.invoke_llm",
                   side_effect=lambda s, u, **k: cap.update(user=u) or "MOCK"), \
             patch("builtins.print", lambda *a, **k: None):
            out = ProductAnalystAgent()(state)
        assert out["agent_metrics"]["product_analyst"]["status"] == "success"
        u = cap["user"]
        # Many gaps collapsed to a single compact line each
        assert u.count("no signal (potential cross-sell gap)") >= 10
        # The active POS Collection (108) line keeps full detail
        pos_line = next(l for l in u.splitlines() if l.startswith("- POS Collection (108"))
        assert "Account Mapping=" in pos_line and "no signal" not in pos_line
        # Compaction must materially shrink the breakdown vs a full dump
        assert "Debit Volume=₺" in u  # active lines still present
        assert u.count("Account Mapping=") <= 3  # only the genuinely active ones

    def test_product_analyst_prompt_carries_catalog(self):
        import builtins
        from unittest.mock import patch
        from mock_data import get_mizan_df
        from agents.data_ingestion import data_ingestion_agent

        state = {"tax_id": "x", "company_name": "T", "sector": "Trading",
                 "mizan_data": get_mizan_df().to_dict(orient="records"),
                 "retry_count": 0, "db_product_flags": {}, "db_financial_metrics": {},
                 "agent_metrics": {}, "execution_timeline": [], "error_log": []}
        state.update(data_ingestion_agent(state))
        cap = {}
        def fake_llm(system_prompt, user_prompt, **kw):
            cap["system"], cap["user"] = system_prompt, user_prompt
            return "MOCK"
        real_print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            with patch("agents.product_analyst.invoke_llm", side_effect=fake_llm):
                from agents.product_analyst import product_analyst_agent
                product_analyst_agent(state)
        finally:
            builtins.print = real_print
        assert "ING PRODUCT CATALOG" in cap["user"]
        assert "PRODUCT-CATALOG SUGGESTION POLICY" in cap["system"]
        assert "YP means foreign currency" in cap["system"]


# ============================================================================
# NACE code mapping + DB-based sector verification
# ============================================================================

class TestNaceMapping:
    def test_exact_ito_code_lookup(self):
        from sector_analysis import sector_from_nace
        assert sector_from_nace("47.11.01") == "Retail"
        assert sector_from_nace("47.11.02") == "Retail"
        assert sector_from_nace("46.19.01") == "Trading"
        assert sector_from_nace("13.10.03") == "Textile"

    def test_code_normalization(self):
        from sector_analysis import normalize_nace_code, sector_from_nace
        assert normalize_nace_code("471101") == "47.11.01"
        assert normalize_nace_code("47 11 01") == "47.11.01"
        assert sector_from_nace("471102") == "Retail"

    def test_division_fallback(self):
        from sector_analysis import sector_from_nace
        # Codes not in the İTO list resolve via 2-digit NACE division
        assert sector_from_nace("47.99.99") == "Retail"
        assert sector_from_nace("29.10.99") == "Automotive"
        assert sector_from_nace("55.10.99") == "Tourism"

    def test_unknown_or_empty(self):
        from sector_analysis import sector_from_nace
        assert sector_from_nace(None) is None
        assert sector_from_nace("") is None
        assert sector_from_nace("99.99.99") is None

    def test_no_duplicate_codes_across_sectors(self):
        import json
        from pathlib import Path
        from collections import Counter
        path = Path(__file__).parent.parent / "data" / "nace_sector_mapping.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter(
            c for v in data["sectors"].values() for c in v["nace_codes"]
        )
        dups = [c for c, n in counts.items() if n > 1]
        assert not dups, f"NACE codes mapped to multiple sectors: {dups[:10]}"


class TestNaceSectorVerification:
    def _run_agent(self, monkeypatch, nace_code, predicted_sector):
        from agents.db_enrichment import db_enrichment_agent
        snapshot = {
            "found": True, "matched_by": "tax_id",
            "financial_metrics": {}, "financial_period": "202606",
            "product_flags": {}, "nace_code": nace_code,
        }
        monkeypatch.setattr("agents.db_enrichment.ensure_db", lambda: None)
        monkeypatch.setattr(
            "agents.db_enrichment.get_customer_snapshot",
            lambda tax_id, company_name: snapshot,
        )
        return db_enrichment_agent.execute(
            {"tax_id": "x", "company_name": "X", "sector": predicted_sector}
        )

    def test_prediction_confirmed_by_nace(self, monkeypatch):
        out = self._run_agent(monkeypatch, "47.11.02", "Retail + Export")
        v = out["db_meta"]["sector_verification"]
        assert v["nace_sector"] == "Retail"
        assert v["verified"] is True
        assert "sector" not in out  # prediction kept untouched

    def test_prediction_mismatch_logged_not_overridden(self, monkeypatch):
        out = self._run_agent(monkeypatch, "13.10.03", "Construction")  # NACE=Textile
        v = out["db_meta"]["sector_verification"]
        assert v["nace_sector"] == "Textile"
        assert v["verified"] is False
        assert "sector" not in out  # predictor remains the source of truth

    def test_absent_prediction_filled_from_nace(self, monkeypatch):
        out = self._run_agent(monkeypatch, "47.11.02", "General")
        assert out["db_meta"]["sector_verification"]["verified"] is False
        assert out["sector"] == "Retail"  # NACE fills the missing prediction

    def test_no_nace_code_skips_verification(self, monkeypatch):
        out = self._run_agent(monkeypatch, None, "Retail")
        assert out["db_meta"]["sector_verification"]["verified"] is None


# ============================================================================
# Sector analysis (TCMB benchmarks)
# ============================================================================

class TestSectorAnalysis:
    @pytest.fixture(autouse=True)
    def _force_enabled(self, monkeypatch):
        # These tests exercise the comparison logic itself, so force the
        # (optional) sector analysis ON regardless of benchmark freshness.
        monkeypatch.setenv("ENABLE_SECTOR_ANALYSIS", "1")

    def test_sector_matching_en_tr(self):
        from sector_analysis import match_sector
        assert match_sector("Textile") == "Textile"
        assert match_sector("Tekstil / Hazır Giyim") == "Textile"
        assert match_sector("İnşaat Taahhüt") == "Construction"
        assert match_sector("Completely Unknown Sector") == "General"

    def test_textile_subcategories_match_textile_not_manufacturing(self):
        """Specific industry must win over the generic 'Manufacturing' keyword."""
        from sector_analysis import match_sectors
        plain = match_sectors("Textile Manufacturing")
        export = match_sectors("Textile Manufacturing + Export")
        assert plain["primary"] == "Textile"
        assert plain["modifiers"] == []
        assert export["primary"] == "Textile"
        assert export["modifiers"] == ["Export"]

    def test_multi_sector_prediction(self):
        from sector_analysis import match_sectors
        info = match_sectors("Retail + Export")
        assert info["primary"] == "Retail"
        assert info["modifiers"] == ["Export"]
        assert info["is_fallback"] is False
        combo = match_sectors("Food & Beverage + Trading")
        assert combo["primary"] == "Food & Beverage"
        assert "Trading" in combo["secondary"]

    def test_fallback_flag_when_unmatched_or_absent(self):
        from sector_analysis import match_sectors
        assert match_sectors("Quantum Basket Weaving")["is_fallback"] is True
        assert match_sectors("")["is_fallback"] is True
        assert match_sectors(None)["is_fallback"] is True
        assert match_sectors("General")["is_fallback"] is True

    def test_comparison_surfaces_modifiers_and_fallback(self):
        from sector_analysis import compare_company_to_sector
        ratios = {"current_ratio": {"value": 1.2}}
        exporter = compare_company_to_sector(ratios, "Textile Manufacturing + Export")
        assert exporter["is_fallback"] is False
        assert exporter["modifiers"] == ["Export"]
        assert "Cross-border profile" in exporter["markdown"]
        fallback = compare_company_to_sector(ratios, "Unknown Sector")
        assert fallback["is_fallback"] is True
        assert "Sector prediction absent or unmatched" in fallback["markdown"]

    def test_comparison_rows_and_markdown(self):
        from sector_analysis import compare_company_to_sector
        ratios = {
            "current_ratio": {"value": 1.1}, "quick_ratio": {"value": 0.7},
            "debt_to_equity": {"value": 2.4}, "gross_margin": {"value": 16.0},
            "collection_period": {"value": 95},
        }
        cmp = compare_company_to_sector(ratios, "Manufacturing")
        assert cmp["matched_sector"] == "Manufacturing"
        assert len(cmp["rows"]) == 5
        assert "TCMB SECTOR BENCHMARK COMPARISON" in cmp["markdown"]
        # debt_to_equity 2.4 vs 1.65 benchmark → worse
        dte = next(r for r in cmp["rows"] if r["metric"] == "debt_to_equity")
        assert dte["assessment"] == "worse than sector"

    def test_missing_ratios_graceful(self):
        from sector_analysis import compare_company_to_sector
        cmp = compare_company_to_sector({}, "Retail")
        assert cmp["rows"] == []
        assert "No comparable benchmark" in cmp["markdown"]

    def test_ratio_bindings_aligned_with_local_db_metrics(self):
        """Every DB metric name in _RATIO_BINDINGS must exist in
        local_db.TR_EN_METRIC_MAP (the EN names the snapshot produces)."""
        from sector_analysis import _RATIO_BINDINGS
        from local_db import TR_EN_METRIC_MAP
        en_names = set(TR_EN_METRIC_MAP.values())
        for bench_key, ratio_key, db_metric_en, unit, direction in _RATIO_BINDINGS:
            if db_metric_en is not None:
                assert db_metric_en in en_names, (
                    f"binding '{bench_key}' references unknown DB metric "
                    f"'{db_metric_en}'"
                )
            # every binding needs at least one company-value source
            assert ratio_key or db_metric_en

    def test_db_metrics_fill_mizan_gaps(self):
        """Net margin & leverage have no Mizan ratio — DB must fill them."""
        from sector_analysis import compare_company_to_sector
        ratios = {"current_ratio": {"value": 1.2}}
        db_metrics = {
            "Net Profit Margin (%)": 5.4,
            "Total Liabilities / Total Assets": 0.70,  # EDW stores ratios → ×100
            "Current Ratio": 1.9,  # must NOT override the Mizan value
        }
        cmp = compare_company_to_sector(ratios, "Manufacturing", db_metrics=db_metrics)
        by_metric = {r["metric"]: r for r in cmp["rows"]}
        # Mizan wins when both sources exist
        cur = by_metric["current_ratio"]
        assert cur["company"] == 1.2 and cur["source"] == "mizan"
        # DB-only metrics appear with bank_db source
        net = by_metric["Net Profit Margin (%)"]
        assert net["company"] == 5.4 and net["source"] == "bank_db"
        lev = by_metric["Total Liabilities / Total Assets"]
        assert lev["company"] == 70.0 and lev["source"] == "bank_db"
        assert "Bank DB (latest)" in cmp["markdown"]

    def test_db_ratio_metrics_fixed_scale(self):
        """EDW convention (confirmed): non-'(%)' ratio metrics are stored
        as plain ratios (0.72) → fixed ×100; '(%)' metrics are untouched."""
        from sector_analysis import compare_company_to_sector
        cmp = compare_company_to_sector(
            {}, "Manufacturing",
            db_metrics={
                "Total Liabilities / Total Assets": 0.72,
                "Total Financial Debt / Total Liabilities": 0.34,
                "Net Profit Margin (%)": 7.5,   # already a percent — no scaling
            },
        )
        by_metric = {r["metric"]: r for r in cmp["rows"]}
        assert by_metric["Total Liabilities / Total Assets"]["company"] == 72.0
        assert by_metric["Total Financial Debt / Total Liabilities"]["company"] == 34.0
        assert by_metric["Net Profit Margin (%)"]["company"] == 7.5
        # Deterministic — NOT threshold-based: a ratio metric is always
        # scaled, even when its raw value exceeds the old 1.5 heuristic cutoff
        cmp2 = compare_company_to_sector(
            {}, "Manufacturing",
            db_metrics={"Total Liabilities / Total Assets": 1.8},  # leveraged firm
        )
        lev2 = next(r for r in cmp2["rows"]
                    if r["metric"] == "Total Liabilities / Total Assets")
        assert lev2["company"] == 180.0


class TestSectorAnalysisOptional:
    """Sector analysis is OPTIONAL — gated by ENABLE_SECTOR_ANALYSIS so a
    stale benchmark JSON can be skipped without code changes."""

    def test_force_disabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SECTOR_ANALYSIS", "0")
        from sector_analysis import is_sector_analysis_enabled, compare_company_to_sector
        assert is_sector_analysis_enabled() is False
        cmp = compare_company_to_sector({"current_ratio": {"value": 1.2}}, "Manufacturing")
        assert cmp["enabled"] is False
        assert cmp["rows"] == [] and cmp["markdown"] == ""

    def test_force_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SECTOR_ANALYSIS", "1")
        from sector_analysis import is_sector_analysis_enabled, compare_company_to_sector
        assert is_sector_analysis_enabled() is True
        cmp = compare_company_to_sector({"current_ratio": {"value": 1.2}}, "Manufacturing")
        assert cmp["enabled"] is True and cmp["rows"]

    def test_auto_follows_benchmark_freshness(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SECTOR_ANALYSIS", "auto")
        import sector_analysis as sa
        # auto + stale data (needs_refresh=true) -> disabled
        monkeypatch.setattr(sa, "load_benchmarks",
                            lambda *a, **k: {"metadata": {"needs_refresh": True}})
        assert sa.is_sector_analysis_enabled() is False
        # auto + fresh data -> enabled
        monkeypatch.setattr(sa, "load_benchmarks",
                            lambda *a, **k: {"metadata": {"needs_refresh": False}})
        assert sa.is_sector_analysis_enabled() is True

    def test_strategist_omits_section_when_disabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SECTOR_ANALYSIS", "0")
        import builtins
        from unittest.mock import patch
        state = {
            "tax_id": "x", "company_name": "Test Co", "sector": "Manufacturing",
            "financial_ratios": {"current_ratio": {"value": 1.1}, "donem_context": {}},
            "network_data": {"stats": {}, "nodes": []},
            "product_signals": {"POS Collection (108)": {"volume": 8e7, "balance": 2e6}},
            "db_product_flags": {}, "db_financial_metrics": {},
        }
        cap = {}
        def fake_llm(system_prompt, user_prompt, **kw):
            cap["user"] = user_prompt
            return "MOCK"
        real_print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            with patch("agents.strategist.invoke_llm", side_effect=fake_llm):
                from agents.strategist import sales_strategist_agent
                out = sales_strategist_agent(state)
        finally:
            builtins.print = real_print
        assert out["agent_metrics"]["strategist"]["status"] == "success"
        assert "TCMB SECTOR BENCHMARK COMPARISON" not in cap["user"]
