"""
LangGraph Agent Orchestration
===============================
Wires agents into a LangGraph StateGraph with:
- Verifier retry loop
- Network mapper for entity graph
- Strategist for final sales report

Architecture:
  START → data_ingestion → quant_analyst → verifier
                                            ↓ (conditional)
                              approved → network_mapper → strategist → END
                              rejected → quant_analyst (loop back)
"""

import logging
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

from agents.state import SwarmState
from agents.data_ingestion import data_ingestion_agent
from agents.quant_analyst import quant_analyst_agent
from agents.verifier import verifier_agent, should_retry_or_continue
from agents.network_mapper import network_mapper_agent
from agents.strategist import sales_strategist_agent

logger = logging.getLogger("swarm.graph")





def _route_after_verification(state: SwarmState) -> str:
    """Router: approved → fan_out (parallel), rejected → quant_analyst (retry)."""
    return should_retry_or_continue(state)


def build_swarm_graph() -> StateGraph:
    """
    Build and compile the multi-agent swarm graph.

    Architecture:
    ┌───────────────────────────────────────────────────┐
    │  START                                             │
    │    ↓                                               │
    │  [data_ingestion] ── Standardize Mizan data        │
    │    ↓                                               │
    │  [quant_analyst] ─── Calculate financial ratios ◄─┐│
    │    ↓                                              ││
    │  [verifier] ──────── Check account codes + math   ││
    │    ↓                                              ││
    │  {conditional} ───── rejected? ───────────────────┘│
    │    ↓ approved                                      │
    │  [network_mapper] ── Build entity graph            │
    │    ↓                                               │
    │  [strategist] ────── Generate sales strategy       │
    │    ↓                                               │
    │  END                                               │
    └───────────────────────────────────────────────────┘
    """

    builder = StateGraph(SwarmState)

    # Nodes
    builder.add_node("data_ingestion", data_ingestion_agent)
    builder.add_node("quant_analyst", quant_analyst_agent)
    builder.add_node("verifier", verifier_agent)
    builder.add_node("network_mapper", network_mapper_agent)
    builder.add_node("strategist", sales_strategist_agent)

    # Sequential: START → data_ingestion → quant_analyst → verifier
    builder.add_edge(START, "data_ingestion")
    builder.add_edge("data_ingestion", "quant_analyst")
    builder.add_edge("quant_analyst", "verifier")

    # Verifier conditional: approved → network_mapper, rejected → retry
    builder.add_conditional_edges(
        "verifier",
        _route_after_verification,
        {
            "network_mapper": "network_mapper",
            "quant_analyst": "quant_analyst",
        }
    )

    # network_mapper → strategist → END
    builder.add_edge("network_mapper", "strategist")
    builder.add_edge("strategist", END)

    graph = builder.compile()

    logger.info("[Graph] Swarm graph compiled successfully!")
    logger.info(f"[Graph] Nodes: {list(builder.nodes.keys())}")

    return graph


# Pre-build the graph at module level
swarm_graph = build_swarm_graph()
