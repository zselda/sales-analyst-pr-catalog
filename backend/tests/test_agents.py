"""
Unit Tests — Individual Agent Tests
=====================================
Tests each agent in isolation using mock data and mock LLM responses.
Does NOT require a running LLM — all LLM calls are mocked.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mock_data import get_mizan_df, get_transactions_df


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_mizan_data():
    """Raw mizan data as list of dicts."""
    return get_mizan_df().to_dict(orient="records")


@pytest.fixture
def mock_transactions_data():
    """Raw transaction data as list of dicts."""
    return get_transactions_df().to_dict(orient="records")


@pytest.fixture
def base_state(mock_mizan_data):
    """Full initial state with all data loaded."""
    return {
        "tax_id": "1234567890",
        "company_name": "Test Company",
        "sector": "General",
        "mizan_data": mock_mizan_data,
        "standardized_mizan": None,
        "financial_ratios": None,
        "verification_status": None,
        "verification_errors": "",
        "retry_count": 0,
        "network_data": None,
        "strategy_report": None,
        "chat_history": [],
        "chat_response": None,
        "agent_metrics": {},
        "execution_timeline": [],
        "pipeline_start_time": None,
        "error_log": [],
    }


@pytest.fixture
def standardized_state(base_state):
    """State after data_ingestion has run."""
    from agents.data_ingestion import data_ingestion_agent
    result = data_ingestion_agent(base_state)
    base_state.update(result)
    return base_state


# ============================================================================
# Agent 1: Data Ingestion Tests
# ============================================================================

class TestDataIngestion:
    def test_standardizes_accounts(self, base_state):
        from agents.data_ingestion import data_ingestion_agent
        result = data_ingestion_agent(base_state)

        mizan = result["standardized_mizan"]
        assert isinstance(mizan, list)
        assert len(mizan) > 0

        # Check classification fields exist
        first = mizan[0]
        assert "category" in first
        assert "net_balance" in first
        assert "is_sub_account" in first
        assert "main_account" in first

    def test_classifies_account_categories(self, base_state):
        from agents.data_ingestion import data_ingestion_agent
        result = data_ingestion_agent(base_state)
        mizan = result["standardized_mizan"]

        categories = set(row["category"] for row in mizan)
        assert "Current Assets" in categories
        assert "Revenue" in categories
        assert "Expenses" in categories
        assert "Equity" in categories

    def test_handles_empty_data(self):
        from agents.data_ingestion import data_ingestion_agent
        result = data_ingestion_agent({"mizan_data": [], "agent_metrics": {}, "execution_timeline": []})
        assert result["standardized_mizan"] == []

    def test_records_metrics(self, base_state):
        from agents.data_ingestion import data_ingestion_agent
        result = data_ingestion_agent(base_state)
        assert "agent_metrics" in result
        assert "data_ingestion" in result["agent_metrics"]
        metrics = result["agent_metrics"]["data_ingestion"]
        assert metrics["status"] == "success"
        assert metrics["execution_time_ms"] >= 0


# ============================================================================
# Agent 2: Quantitative Analyst Tests
# ============================================================================

class TestQuantAnalyst:
    @patch("agents.quant_analyst.invoke_llm", return_value="Mock LLM interpretation.")
    def test_calculates_ratios(self, mock_llm, standardized_state):
        from agents.quant_analyst import quant_analyst_agent
        result = quant_analyst_agent(standardized_state)

        ratios = result["financial_ratios"]
        assert "gross_margin" in ratios
        assert "current_ratio" in ratios
        assert "debt_to_equity" in ratios
        assert "financial_expense_ratio" in ratios
        assert "pos_commission_ratio" in ratios

    @patch("agents.quant_analyst.invoke_llm", return_value="Mock LLM interpretation.")
    def test_ratio_values_in_bounds(self, mock_llm, standardized_state):
        from agents.quant_analyst import quant_analyst_agent
        result = quant_analyst_agent(standardized_state)

        ratios = result["financial_ratios"]
        assert 0 < ratios["gross_margin"]["value"] < 100
        assert ratios["current_ratio"]["value"] > 0
        assert ratios["debt_to_equity"]["value"] > 0

    @patch("agents.quant_analyst.invoke_llm", return_value="Mock interpretation.")
    def test_includes_account_codes(self, mock_llm, standardized_state):
        from agents.quant_analyst import quant_analyst_agent
        result = quant_analyst_agent(standardized_state)

        gm = result["financial_ratios"]["gross_margin"]
        assert "600" in gm["accounts_used"]
        assert "620" in gm["accounts_used"]

    @patch("agents.quant_analyst.invoke_llm", return_value="Mock interpretation.")
    def test_hesap_kodu_aciklama_present(self, mock_llm, standardized_state):
        """Test that raw_values include hesap_kodu_aciklama citations."""
        from agents.quant_analyst import quant_analyst_agent
        result = quant_analyst_agent(standardized_state)

        gm_raw = result["financial_ratios"]["gross_margin"]["raw_values"]
        assert isinstance(gm_raw["revenue_600"], dict)
        assert gm_raw["revenue_600"]["hesap_kodu"] == "600"
        assert gm_raw["revenue_600"]["hesap_kodu_aciklama"] == "YURTİÇİ SATIŞLAR"

        assert isinstance(gm_raw["cogs_620"], dict)
        assert gm_raw["cogs_620"]["hesap_kodu"] == "620"
        assert gm_raw["cogs_620"]["hesap_kodu_aciklama"] == "SATILAN MALIN MALİYETİ (-)"


# ============================================================================
# Agent 3: Verifier Tests
# ============================================================================

class TestVerifier:
    @patch("agents.verifier.invoke_llm", return_value="APPROVED. All ratios valid.")
    def test_approves_correct_ratios(self, mock_llm, standardized_state):
        from agents.quant_analyst import quant_analyst_agent
        from agents.verifier import verifier_agent

        with patch("agents.quant_analyst.invoke_llm", return_value="Mock."):
            qa_result = quant_analyst_agent(standardized_state)
        standardized_state.update(qa_result)

        result = verifier_agent(standardized_state)
        assert result["verification_status"] == "approved"

    def test_rejects_missing_ratios(self):
        from agents.verifier import verifier_agent
        state = {
            "financial_ratios": {"error": "No data"},
            "retry_count": 0,
            "agent_metrics": {},
            "execution_timeline": [],
        }
        result = verifier_agent(state)
        assert result["verification_status"] == "rejected"

    def test_retry_routing(self):
        from agents.verifier import should_retry_or_continue
        assert should_retry_or_continue({"verification_status": "approved", "retry_count": 1}) == "network_mapper"
        assert should_retry_or_continue({"verification_status": "rejected", "retry_count": 0}) == "quant_analyst"
        assert should_retry_or_continue({"verification_status": "rejected", "retry_count": 3}) == "network_mapper"


# ============================================================================
# Agent 5: Network Mapper Tests
# ============================================================================

class TestNetworkMapper:
    def test_builds_graph(self, standardized_state):
        from agents.network_mapper import network_mapper_agent
        result = network_mapper_agent(standardized_state)

        network = result["network_data"]
        assert len(network["nodes"]) > 0
        assert len(network["edges"]) > 0

    def test_correct_node_counts(self, standardized_state):
        from agents.network_mapper import network_mapper_agent
        result = network_mapper_agent(standardized_state)

        stats = result["network_data"]["stats"]
        assert stats["customer_count"] == 5
        assert stats["supplier_count"] == 5
        assert stats["bank_count"] == 3
        assert stats["total_nodes"] == 14  # 5 + 5 + 3 + 1 target

    def test_node_types(self, standardized_state):
        from agents.network_mapper import network_mapper_agent
        result = network_mapper_agent(standardized_state)

        types = set(n["type"] for n in result["network_data"]["nodes"])
        assert "target" in types
        assert "customer" in types
        assert "supplier" in types
        assert "bank" in types


# ============================================================================
# Agent 6: Strategist Tests
# ============================================================================

class TestStrategist:
    def _make_strategist_state(self, standardized_state):
        """Helper: set up upstream data for strategist tests."""
        standardized_state["financial_ratios"] = {
            "gross_margin": {"value": 30.0, "unit": "%", "formula": "test", "accounts_used": ["600", "620"],
                             "raw_values": {
                                 "revenue_600": {"value": 1000000, "hesap_kodu": "600", "hesap_kodu_aciklama": "YURTİÇİ SATIŞLAR"},
                                 "cogs_620": {"value": 700000, "hesap_kodu": "620", "hesap_kodu_aciklama": "SATILAN MALIN MALİYETİ (-)"}
                             }},
            "operating_margin": {"value": 15.0, "unit": "%", "formula": "test", "accounts_used": ["600", "620", "630", "631", "632"],
                                  "raw_values": {}},
            "current_ratio": {"value": 2.0, "unit": "x", "formula": "test", "accounts_used": [], "raw_values": {}},
            "quick_ratio": {"value": 1.5, "unit": "x", "formula": "test", "accounts_used": [],
                            "raw_values": {
                                "received_checks_101": {"value": 50000, "hesap_kodu": "101", "hesap_kodu_aciklama": "ALINAN ÇEKLER"},
                                "given_checks_103": {"value": 30000, "hesap_kodu": "103", "hesap_kodu_aciklama": "VERİLEN ÇEKLER VE ÖDEME EMİRLERİ"}
                            }},
            "collection_period": {"value": 45, "unit": "days", "formula": "test", "accounts_used": ["120", "121", "600"],
                                   "raw_values": {
                                       "trade_receivables_120": {"value": 80000, "hesap_kodu": "120", "hesap_kodu_aciklama": "ALICILAR"},
                                       "trade_receivables_121": {"value": 20000, "hesap_kodu": "121", "hesap_kodu_aciklama": "ALACAK SENETLERİ"},
                                       "trade_receivables": 100000,
                                       "revenue_600": {"value": 1000000, "hesap_kodu": "600", "hesap_kodu_aciklama": "YURTİÇİ SATIŞLAR"}
                                   }},
            "payment_period": {"value": 30, "unit": "days", "formula": "test", "accounts_used": ["320", "321", "620"],
                                "raw_values": {
                                    "trade_payables_320": {"value": 50000, "hesap_kodu": "320", "hesap_kodu_aciklama": "SATICILAR"},
                                    "trade_payables_321": {"value": 10000, "hesap_kodu": "321", "hesap_kodu_aciklama": "BORÇ SENETLERİ"},
                                    "trade_payables": 60000,
                                    "cogs_620": {"value": 700000, "hesap_kodu": "620", "hesap_kodu_aciklama": "SATILAN MALIN MALİYETİ (-)"}
                                }},
            "debt_to_equity": {"value": 1.5, "unit": "x", "formula": "test", "accounts_used": [], "raw_values": {}},
            "bank_debt_ratio": {"value": 40.0, "unit": "%", "formula": "test", "accounts_used": ["300", "309", "400"],
                                 "raw_values": {
                                     "banka_kredileri_kv_300": {"value": 200000, "hesap_kodu": "300", "hesap_kodu_aciklama": "BANKA KREDİLERİ"},
                                     "banka_kredileri_uv_400": {"value": 100000, "hesap_kodu": "400", "hesap_kodu_aciklama": "BANKA KREDİLERİ"},
                                     "diger_mali_borclar_309": {"value": 30000, "hesap_kodu": "309", "hesap_kodu_aciklama": "DİĞER MALİ BORÇLAR"},
                                     "total_bank_loans": 330000,
                                     "total_liabilities": 825000
                                 }},
            "financial_expense_ratio": {"value": 3.5, "unit": "%", "formula": "test", "accounts_used": ["780", "600"],
                                         "raw_values": {
                                             "finansman_giderleri_780": {"value": 35000, "hesap_kodu": "780", "hesap_kodu_aciklama": "FİNANSMAN GİDERLERİ"},
                                             "revenue_600": {"value": 1000000, "hesap_kodu": "600", "hesap_kodu_aciklama": "YURTİÇİ SATIŞLAR"}
                                         }},
            "pos_commission_ratio": {"value": 1.5, "unit": "%", "formula": "test", "accounts_used": ["780.01", "600"],
                                      "raw_values": {
                                          "pos_komisyon_780_01": {"value": 15000, "hesap_kodu": "780.01", "hesap_kodu_aciklama": "POS KOMİSYON GİDERLERİ"},
                                          "revenue_600": {"value": 1000000, "hesap_kodu": "600", "hesap_kodu_aciklama": "YURTİÇİ SATIŞLAR"}
                                      }},
            "competitor_banks": {
                "102": [{"name": "Akbank", "balance": 50000, "share_pct": 60.0}],
                "300": [{"name": "Garanti", "balance": 100000, "share_pct": 50.0}],
                "400": []
            },
            "llm_interpretation": "Test interpretation.",
        }
        standardized_state["network_data"] = {
            "nodes": [{"label": "Test Corp", "type": "customer", "balance": 100000}],
            "edges": [],
            "stats": {"total_receivables": 100000, "total_payables": 50000},
        }
        return standardized_state

    def test_calls_llm_with_local_calculations(self, standardized_state):
        """Test that strategist sends all local calculations to LLM and returns LLM report."""
        from agents.strategist import sales_strategist_agent

        state = self._make_strategist_state(standardized_state)
        mock_report = "# Strategy Report\nThis is a comprehensive LLM-generated report with reasoning about the company's financial health."

        with patch("agents.strategist.invoke_llm", return_value=mock_report) as mock_llm:
            result = sales_strategist_agent(state)

        # Verify LLM was called
        mock_llm.assert_called_once()

        # Verify the prompt contains real calculated values
        call_args = mock_llm.call_args
        prompt = call_args[0][1]  # second positional arg is the prompt
        assert "30.0%" in prompt, "Prompt should contain gross margin value"
        assert "YURTİÇİ SATIŞLAR" in prompt, "Prompt should contain hesap kodu açıklama"
        assert "₺1,000,000" in prompt or "1,000,000" in prompt, "Prompt should contain revenue"
        assert "Akbank" in prompt, "Prompt should contain competitor bank name"
        assert "POS Commission" in prompt, "Prompt should contain POS data"

        # Verify report is the LLM output
        report = result["strategy_report"]
        assert report == mock_report, "Should return exactly what LLM generated"

    def test_prompt_includes_all_sections(self, standardized_state):
        """Verify the prompt sent to LLM includes all data sections."""
        from agents.strategist import sales_strategist_agent

        state = self._make_strategist_state(standardized_state)

        with patch("agents.strategist.invoke_llm", return_value="LLM report output") as mock_llm:
            sales_strategist_agent(state)

        prompt = mock_llm.call_args[0][1]
        assert "KEY PERFORMANCE INDICATORS" in prompt
        assert "RAW ACCOUNT BALANCES" in prompt
        assert "COMPETITOR BANK WALLET SHARE" in prompt
        assert "COMMERCIAL NETWORK" in prompt
        assert "7 sections" in prompt
