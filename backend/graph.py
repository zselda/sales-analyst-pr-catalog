"""
LangGraph Agent Orchestration
===============================
Wires agents into a LangGraph StateGraph with:
- Verifier retry loop
- Product analyst for banking product signal extraction
- Network mapper for entity graph
- Strategist for final sales report

Architecture:
  START → data_ingestion → quant_analyst → verifier
                                            ↓ (conditional)
                              approved → product_analyst → network_mapper → strategist → END
                              rejected → quant_analyst (loop back)
"""

import logging
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

from agents.state import SwarmState
from agents.data_ingestion import data_ingestion_agent
from agents.quant_analyst import quant_analyst_agent
from agents.verifier import verifier_agent, should_retry_or_continue
from agents.product_analyst import product_analyst_agent
from agents.network_mapper import network_mapper_agent
from agents.strategist import sales_strategist_agent
from agents.translator import translator_agent

logger = logging.getLogger("swarm.graph")


def _route_after_verification(state: SwarmState) -> str:
    """Router: approved → product_analyst (then network_mapper), rejected → quant_analyst (retry)."""
    return should_retry_or_continue(state)


def build_swarm_graph() -> StateGraph:
    """
    Build and compile the multi-agent swarm graph.

    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │  START                                                     │
    │    ↓                                                       │
    │  [data_ingestion] ── Standardize Mizan data                │
    │    ↓                                                       │
    │  [quant_analyst] ─── Calculate financial ratios ◄─────────┐│
    │    ↓                                                      ││
    │  [verifier] ──────── Check account codes + math           ││
    │    ↓                                                      ││
    │  {conditional} ───── rejected? ───────────────────────────┘│
    │    ↓ approved                                              │
    │  [product_analyst] ─ Extract product signals               │
    │    ↓                                                       │
    │  [network_mapper] ── Build entity graph                    │
    │    ↓                                                       │
    │  [strategist] ────── Generate ING Bank sales strategy      │
    │    ↓                                                       │
    │  END                                                       │
    └───────────────────────────────────────────────────────────┘
    """

    builder = StateGraph(SwarmState)

    # Nodes
    builder.add_node("data_ingestion", data_ingestion_agent)
    builder.add_node("quant_analyst", quant_analyst_agent)
    builder.add_node("verifier", verifier_agent)
    builder.add_node("product_analyst", product_analyst_agent)
    builder.add_node("network_mapper", network_mapper_agent)
    builder.add_node("strategist", sales_strategist_agent)

    # Sequential: START → data_ingestion → quant_analyst → verifier
    builder.add_edge(START, "data_ingestion")
    builder.add_edge("data_ingestion", "quant_analyst")
    builder.add_edge("quant_analyst", "verifier")

    # Verifier conditional: approved → product_analyst, rejected → retry
    builder.add_conditional_edges(
        "verifier",
        _route_after_verification,
        {
            "product_analyst": "product_analyst",
            "quant_analyst": "quant_analyst",
        }
    )

    # product_analyst → network_mapper → strategist → END
    builder.add_edge("product_analyst", "network_mapper")
    builder.add_edge("network_mapper", "strategist")
    builder.add_edge("strategist", END)

    graph = builder.compile()

    logger.info("[Graph] Swarm graph compiled successfully!")
    logger.info(f"[Graph] Nodes: {list(builder.nodes.keys())}")

    return graph


# Pre-build the graph at module level
swarm_graph = build_swarm_graph()


def build_standalone_graph(generate_turkish: bool = False):
    """
    Build graph for standalone pipeline execution.

    Adds an optional translator node after strategist when generate_turkish=True.

    Architecture (with translation):
      START → data_ingestion → quant_analyst → verifier
                                                ↓ (conditional)
                                  approved → product_analyst → network_mapper → strategist → translator → END
                                  rejected → quant_analyst (loop back)

    Architecture (without translation):
      Same as build_swarm_graph() — strategist → END
    """
    builder = StateGraph(SwarmState)

    # Nodes
    builder.add_node("data_ingestion", data_ingestion_agent)
    builder.add_node("quant_analyst", quant_analyst_agent)
    builder.add_node("verifier", verifier_agent)
    builder.add_node("product_analyst", product_analyst_agent)
    builder.add_node("network_mapper", network_mapper_agent)
    builder.add_node("strategist", sales_strategist_agent)

    # Sequential: START → data_ingestion → quant_analyst → verifier
    builder.add_edge(START, "data_ingestion")
    builder.add_edge("data_ingestion", "quant_analyst")
    builder.add_edge("quant_analyst", "verifier")

    # Verifier conditional
    builder.add_conditional_edges(
        "verifier",
        _route_after_verification,
        {
            "product_analyst": "product_analyst",
            "quant_analyst": "quant_analyst",
        }
    )

    builder.add_edge("product_analyst", "network_mapper")
    builder.add_edge("network_mapper", "strategist")

    if generate_turkish:
        builder.add_node("translator", translator_agent)
        builder.add_edge("strategist", "translator")
        builder.add_edge("translator", END)
        logger.info("[Graph] Standalone graph compiled WITH translator (Turkish)")
    else:
        builder.add_edge("strategist", END)
        logger.info("[Graph] Standalone graph compiled WITHOUT translator (English only)")

    graph = builder.compile()
    logger.info(f"[Graph] Standalone nodes: {list(builder.nodes.keys())}")
    return graph
