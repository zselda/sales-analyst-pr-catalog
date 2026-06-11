"""
Agent 5: Network Mapper Agent
================================
Extracts B2B relationships from Mizan sub-accounts (120.xx and 320.xx)
and builds an in-memory directed graph using NetworkX.
 
Refactored to use BaseAgent for tracing and error isolation.
"""
 
import logging
import networkx as nx
import pandas as pd
from agents.base import BaseAgent
 
logger = logging.getLogger("swarm.agents.network_mapper")
 
 
class NetworkMapperAgent(BaseAgent):
    name = "network_mapper"
    description = "Build commercial network graph from Mizan sub-accounts"
    required_inputs = ["standardized_mizan"]
    output_keys = ["network_data"]
 
    def execute(self, state: dict) -> dict:
        """Build a directed commercial network graph from Mizan data."""
 
        standardized = state.get("standardized_mizan", [])
        if not standardized:
            return {"network_data": {"nodes": [], "edges": [], "stats": {}}}
 
        df = pd.DataFrame(standardized)
        G = nx.DiGraph()
 
        # ── CENTER NODE: Target Company ──
        tax_id = state.get("tax_id", "1234567890")
        G.add_node(tax_id, **{
            "label": "Target Company",
            "type": "target",
            "color": "#FFD700",
            "size": 60,
        })
 
        # ── CUSTOMER NODES (120.xx) ──
        customers_df = df[df["account_code"].str.startswith("120")]
        # Filter to leaf accounts only (no children)
        cust_codes = customers_df["account_code"].tolist()
        cust_leaves = [c for c in cust_codes if not any(
            o.startswith(c) and len(o) > len(c) for o in cust_codes
        )]
        customers_df = customers_df[customers_df["account_code"].isin(cust_leaves)]
 
        for _, row in customers_df.iterrows():
            name = str(row["account_name"]).strip()
            # Remove common prefixes if present
            for prefix in ["Alicilar - ", "ALICILAR - ", "Alıcılar - "]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
            node_id = f"cust_{row['account_code']}"
            # Use balance_debit if available, else debit
            balance = float(row.get("balance_debit", 0) or row.get("debit", 0) or 0)
 
            G.add_node(node_id, **{
                "label": name, "type": "customer", "color": "#4CAF50",
                "size": max(20, min(50, balance / 200000)),
                "balance": balance, "account_code": row["account_code"],
            })
            G.add_edge(node_id, tax_id, **{
                "weight": balance, "label": f"₺{balance:,.0f}",
                "type": "receivable", "color": "#4CAF50",
            })
 
        # ── SUPPLIER NODES (320.xx) ──
        suppliers_df = df[df["account_code"].str.startswith("320")]
        # Filter to leaf accounts only
        supp_codes = suppliers_df["account_code"].tolist()
        supp_leaves = [c for c in supp_codes if not any(
            o.startswith(c) and len(o) > len(c) for o in supp_codes
        )]
        suppliers_df = suppliers_df[suppliers_df["account_code"].isin(supp_leaves)]
 
        for _, row in suppliers_df.iterrows():
            name = str(row["account_name"]).strip()
            for prefix in ["Saticilar - ", "SATICILAR - ", "Satıcılar - "]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
            node_id = f"supp_{row['account_code']}"
            balance = float(row.get("balance_credit", 0) or row.get("credit", 0) or 0)
 
            G.add_node(node_id, **{
                "label": name, "type": "supplier", "color": "#F44336",
                "size": max(20, min(50, balance / 200000)),
                "balance": balance, "account_code": row["account_code"],
            })
            G.add_edge(tax_id, node_id, **{
                "weight": balance, "label": f"₺{balance:,.0f}",
                "type": "payable", "color": "#F44336",
            })
 
        # ── BANK NODES (102.xx) ──
        banks_df = df[df["account_code"].str.startswith("102")]
        # Filter to leaf accounts only
        bank_codes = banks_df["account_code"].tolist()
        bank_leaves = [c for c in bank_codes if not any(
            o.startswith(c) and len(o) > len(c) for o in bank_codes
        )]
        banks_df = banks_df[banks_df["account_code"].isin(bank_leaves)]
 
        for _, row in banks_df.iterrows():
            name = str(row["account_name"]).strip()
            for prefix in ["Bankalar - ", "BANKALAR - ", "BANKALAR TL - "]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
            node_id = f"bank_{row['account_code']}"
            balance = float(row.get("balance_debit", 0) or row.get("debit", 0) or 0)
 
            G.add_node(node_id, **{
                "label": name, "type": "bank", "color": "#2196F3",
                "size": max(25, min(50, balance / 400000)),
                "balance": balance, "account_code": row["account_code"],
            })
            G.add_edge(tax_id, node_id, **{
                "weight": balance, "label": f"₺{balance:,.0f}",
                "type": "deposit", "color": "#2196F3",
            })
 
        # ── Export graph data ──
        nodes = [{
            "id": nid, "label": attrs.get("label", nid),
            "type": attrs.get("type", "unknown"), "color": attrs.get("color", "#999"),
            "size": attrs.get("size", 30), "balance": attrs.get("balance", 0),
            "account_code": attrs.get("account_code", ""),
        } for nid, attrs in G.nodes(data=True)]
 
        edges = [{
            "source": src, "target": tgt,
            "weight": attrs.get("weight", 0), "label": attrs.get("label", ""),
            "type": attrs.get("type", ""), "color": attrs.get("color", "#999"),
        } for src, tgt, attrs in G.edges(data=True)]
        total_receivables = sum(n["balance"] for n in nodes if n["type"] == "customer")
        total_payables = sum(n["balance"] for n in nodes if n["type"] == "supplier")
        max_customer_ratio = 0
        if total_receivables > 0:
            top_customer = max([n for n in nodes if n["type"] == "customer"], key=lambda x: x["balance"], default=None)
            if top_customer:
                max_customer_ratio = top_customer["balance"] / total_receivables
 
        # 
        max_supplier_ratio = 0
        if total_payables > 0:
            top_supplier = max([n for n in nodes if n["type"] == "supplier"], key=lambda x: x["balance"], default=None)
            if top_supplier:
                max_supplier_ratio = top_supplier["balance"] / total_payables
 
        # %40 ve üzeri bağımlılık varsa Yoğunlaşma Bayrağını (Flag) True yap
        concentration_flag = bool(max_customer_ratio > 0.40 or max_supplier_ratio > 0.40)
 
        
        network = {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "customer_count": len(customers_df),
                "supplier_count": len(suppliers_df),
                "bank_count": len(banks_df),
                "total_receivables": float(customers_df["debit"].sum()),
                "total_payables": float(suppliers_df["credit"].sum()),
                "total_bank_deposits": float(banks_df["debit"].sum()),
                "concentration_flag": concentration_flag, # STRATEJİST İÇİN KRİTİK VERİ
                "max_customer_dependency_ratio": max_customer_ratio,
                "max_supplier_dependency_ratio": max_supplier_ratio
            },
        }
 
        logger.info(f"Network graph: {network['stats']['total_nodes']} nodes, "
                     f"{network['stats']['total_edges']} edges")
 
        return {"network_data": network}
 
 
# Module-level callable for LangGraph
network_mapper_agent = NetworkMapperAgent()
