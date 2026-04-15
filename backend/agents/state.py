"""
LangGraph State Definition
============================
Defines the shared state (TypedDict) that flows through all agents
in the Hunter Swarm pipeline.

Uses Annotated types with reducer functions for keys that receive
concurrent updates from agents.
"""

import operator
from typing import TypedDict, Optional, Any, Annotated


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer: merge two dicts (right overwrites left on conflicts)."""
    if not left:
        return right or {}
    if not right:
        return left or {}
    merged = dict(left)
    merged.update(right)
    return merged


class SwarmState(TypedDict):
    """
    Shared state for the multi-agent swarm pipeline.

    Keys updated by parallel agents use Annotated with reducers:
    - agent_metrics: merged via dict merge (each agent writes its own key)
    - execution_timeline: merged via list append
    - error_log: merged via list append
    """

    # ── Core Data ──
    tax_id: str
    company_name: Optional[str]
    sector: Optional[str]
    mizan_data: Optional[list]
    standardized_mizan: Optional[list]
    donem_info: Optional[dict]
    financial_ratios: Optional[dict]
    verification_status: Optional[str]
    verification_errors: Optional[str]
    retry_count: int
    network_data: Optional[dict]
    strategy_report: Optional[str]
    product_signals: Optional[dict]
    translated_report: Optional[str]
    report_language: Optional[str]
    chat_history: Optional[list]
    chat_response: Optional[str]

    # ── Observability (concurrent-safe with reducers) ──
    agent_metrics: Annotated[dict, merge_dicts]
    execution_timeline: Annotated[list, operator.add]
    pipeline_start_time: Optional[str]
    error_log: Annotated[list, operator.add]
