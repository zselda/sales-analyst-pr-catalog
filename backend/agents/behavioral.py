"""
Agent 4: Behavioral Analyst — Local Pandas + LLM interpretation
 
Refactored to use BaseAgent for tracing and error isolation.
"""
 
import logging
import pandas as pd
import numpy as np
from agents.base import BaseAgent
from llm_config import invoke_llm, BEHAVIORAL_SYSTEM_PROMPT
 
logger = logging.getLogger("swarm.agents.behavioral")
 
 
class BehavioralAnalystAgent(BaseAgent):
    name = "behavioral"
    description = "Analyze transaction patterns, cash flow, and competitor activity"
    required_inputs = ["transactions_data"]
    output_keys = ["behavioral_insights"]
 
    def execute(self, state: dict) -> dict:
        raw = state.get("transactions_data", [])
        # Transactions are optional — return defaults if empty
        if not raw:
            logger.info("No transaction data provided — skipping behavioral analysis")
            return {"behavioral_insights": {
                "cash_flow_crunches": [],
                "competitor_bank_activity": {"risk_flag": "N/A", "total_transactions": 0, "banks_detected": []},
                "collection_mix": {"pos_share_pct": 0},
                "top_outgoing": [],
                "summary_stats": {
                    "total_transactions": 0, "total_inflow": 0, "total_outflow": 0,
                    "unique_counterparties": 0, "date_range": "N/A",
                },
                "llm_interpretation": "No transaction data available for analysis.",
            }}
 
        df = pd.DataFrame(raw)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
 
        # ── 1. Monthly Cash Flow ──
        inc = df[df["Type"] == "Incoming"].groupby("Month")["Amount"].sum()
        out = df[df["Type"] == "Outgoing"].groupby("Month")["Amount"].sum()
        mf = pd.DataFrame({"inflow": inc, "outflow": out}).fillna(0)
        mf["net_flow"] = mf["inflow"] - mf["outflow"]
        monthly_flow_dict = mf.reset_index().to_dict(orient="records")
 
        # ── 2. Top counterparties ──
        def top5(t):
            return (df[df["Type"] == t].groupby("Counterparty_Name")["Amount"]
                    .agg(["sum", "count"]).sort_values("sum", ascending=False)
                    .head(5).reset_index()
                    .rename(columns={"sum": "total_amount", "count": "tx_count"})
                    .to_dict(orient="records"))
        top_in, top_out = top5("Incoming"), top5("Outgoing")
 
        # ── 3. Competitor bank activity ──
        comp_banks = ["Akbank", "Ziraat Bankasi", "Halkbank"]
        ct = df[df["Counterparty_Name"].isin(comp_banks)]
        competitor = {
            "total_transactions": len(ct),
            "total_amount": round(float(ct["Amount"].sum()), 2),
            "banks_detected": ct["Counterparty_Name"].unique().tolist(),
            "frequency_per_bank": (ct.groupby("Counterparty_Name")["Amount"]
                                   .agg(["sum", "count"]).reset_index()
                                   .rename(columns={"sum": "total_amount", "count": "tx_count"})
                                   .to_dict(orient="records")),
            "risk_flag": "HIGH" if len(ct) > 5 else "MODERATE" if len(ct) > 2 else "LOW",
        }
 
        # ── 4. Payment regularity ──
        sp = df[df["Description"].str.contains("Supplier|supplier|Raw material", case=False, na=False)]
        regularity = (sp.groupby("Counterparty_Name")
                      .agg(avg_amount=("Amount", "mean"), std_amount=("Amount", "std"),
                           tx_count=("Amount", "count"), total_amount=("Amount", "sum"))
                      .fillna(0).round(2).reset_index().to_dict(orient="records")) if len(sp) else []
 
        # ── 5. Cash crunch ──
        crunches = []
        for _, row in mf.iterrows():
            if row["outflow"] > 0 and row["net_flow"] < 0:
                sev = abs(row["net_flow"]) / row["outflow"] * 100
                crunches.append({"month": str(row.name), "net_flow": round(float(row["net_flow"]), 2),
                                 "severity_pct": round(sev, 1),
                                 "flag": "CRITICAL" if sev > 30 else "WARNING"})
 
        # ── 6. Collection mix ──
        pos = df[df["Description"].str.contains("POS", case=False, na=False)]
        xfr = df[df["Description"].str.contains("transfer", case=False, na=False)]
        ti = float(df[df["Type"] == "Incoming"]["Amount"].sum())
        mix = {
            "pos_collections": {"count": len(pos), "total": round(float(pos["Amount"].sum()), 2)},
            "bank_transfers": {"count": len(xfr), "total": round(float(xfr["Amount"].sum()), 2)},
            "pos_share_pct": round(float(pos["Amount"].sum()) / ti * 100, 1) if ti else 0,
        }
 
        stats = {
            "total_transactions": len(df),
            "total_inflow": round(float(df[df["Type"] == "Incoming"]["Amount"].sum()), 2),
            "total_outflow": round(float(df[df["Type"] == "Outgoing"]["Amount"].sum()), 2),
            "unique_counterparties": int(df["Counterparty_Name"].nunique()),
            "date_range": f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
        }
 
        logger.info(f"Local: {len(crunches)} crunches, risk={competitor['risk_flag']}")
 
        # ── LLM INTERPRETATION ──
        llm_text = ""
        try:
            crunch_lines = "\n".join(f"  - {c['month']}: ₺{c['net_flow']:,.0f} ({c['flag']})" for c in crunches) or "  None"
            supplier_lines = "\n".join(f"  - {s['Counterparty_Name']}: ₺{s['total_amount']:,.0f}" for s in top_out[:3])
            company_name = state.get("company_name", "Company")
            prompt = (
                f"{company_name} — 6-month transaction analysis:\n\n"
                f"Transactions: {stats['total_transactions']}, "
                f"Inflow: ₺{stats['total_inflow']:,.0f}, Outflow: ₺{stats['total_outflow']:,.0f}\n\n"
                f"Cash Crunches ({len(crunches)} months):\n{crunch_lines}\n\n"
                f"Competitor Banks: {competitor['risk_flag']} risk, "
                f"{competitor['total_transactions']} txns to {', '.join(competitor['banks_detected'])}\n\n"
                f"POS share: {mix['pos_share_pct']}% of collections\n\n"
                f"Top suppliers:\n{supplier_lines}\n\n"
                f"Provide: 1) Key patterns 2) Risk assessment 3) Product recommendations"
            )
            llm_text = invoke_llm(BEHAVIORAL_SYSTEM_PROMPT, prompt, temperature=0.4, max_tokens=1500)
            self.metrics.record_llm_call(tokens=len(llm_text.split()))
            logger.info(f"✅ LLM interpretation: {len(llm_text)} chars")
        except Exception as e:
            logger.warning(f"LLM skipped: {e}")
            llm_text = "LLM interpretation unavailable."
 
        return {"behavioral_insights": {
            "monthly_cash_flow": monthly_flow_dict,
            "top_incoming_counterparties": top_in,
            "top_outgoing_counterparties": top_out,
            "competitor_bank_activity": competitor,
            "supplier_payment_regularity": regularity,
            "cash_flow_crunches": crunches,
            "collection_mix": mix,
            "summary_stats": stats,
            "llm_interpretation": llm_text,
        }}
 
 
# Module-level callable for LangGraph
behavioral_analyst_agent = BehavioralAnalystAgent()
