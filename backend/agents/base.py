"""
Base Agent Infrastructure
============================
Abstract base agent class providing tracing, metrics collection,
error isolation, and lifecycle hooks for all swarm agents.

Every agent inherits from BaseAgent to get:
- Automatic execution timing
- LLM call counting
- Error-isolated execution with graceful degradation
- Input/output validation hooks
- Metrics written back to state
"""

import time
import traceback
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("swarm.agents")


class AgentMetrics:
    """Collects execution metrics for a single agent run."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.llm_calls: int = 0
        self.llm_total_tokens: int = 0
        self.status: str = "pending"
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None

    def start(self):
        self.start_time = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.status = "running"

    def finish(self, status: str = "success", error: Optional[str] = None):
        self.end_time = time.monotonic()
        self.status = status
        self.error = error

    def record_llm_call(self, tokens: int = 0):
        self.llm_calls += 1
        self.llm_total_tokens += tokens

    @property
    def execution_time_ms(self) -> float:
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "execution_time_ms": self.execution_time_ms,
            "llm_calls": self.llm_calls,
            "llm_total_tokens": self.llm_total_tokens,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all swarm agents.

    Subclasses implement `execute()` with their domain logic.
    The base class wraps execution with tracing, metrics, and error isolation.

    Usage:
        class MyAgent(BaseAgent):
            name = "my_agent"
            description = "Does something useful"
            required_inputs = ["mizan_data"]
            output_keys = ["result_key"]

            def execute(self, state: dict) -> dict:
                # domain logic here
                return {"result_key": computed_value}
    """

    name: str = "base_agent"
    description: str = ""
    required_inputs: list[str] = []
    output_keys: list[str] = []

    def __init__(self):
        self.metrics: Optional[AgentMetrics] = None

    def validate_input(self, state: dict) -> list[str]:
        """
        Validate that required input keys exist in state.
        Returns list of missing keys (empty = valid).
        Override for custom validation.
        """
        missing = []
        for key in self.required_inputs:
            val = state.get(key)
            if val is None or (isinstance(val, (list, dict)) and len(val) == 0):
                missing.append(key)
        return missing

    def validate_output(self, output: dict) -> list[str]:
        """
        Validate that output contains expected keys.
        Returns list of missing keys (empty = valid).
        Override for custom validation.
        """
        missing = []
        for key in self.output_keys:
            if key not in output:
                missing.append(key)
        return missing

    @abstractmethod
    def execute(self, state: dict) -> dict:
        """
        Core agent logic. Must return a dict of state updates.
        Subclasses implement this method.
        """
        ...

    def get_fallback_output(self, error: str) -> dict:
        """
        Return a safe fallback output when the agent fails.
        Override for custom fallback behavior.
        """
        return {key: {"error": error} for key in self.output_keys}

    def __call__(self, state: dict) -> dict:
        """
        Execute the agent with full tracing and error isolation.
        This is the entry point called by LangGraph.
        """
        self.metrics = AgentMetrics(self.name)
        self.metrics.start()

        logger.info(f"[{self.name}] Starting execution...")

        # --- Input validation ---
        missing_inputs = self.validate_input(state)
        if missing_inputs:
            warning = f"Missing inputs: {missing_inputs}"
            logger.warning(f"[{self.name}] {warning}")

        # --- Execute with error isolation ---
        try:
            output = self.execute(state)

            # Output validation
            missing_outputs = self.validate_output(output)
            if missing_outputs:
                logger.warning(f"[{self.name}] Missing output keys: {missing_outputs}")

            self.metrics.finish(status="success")
            logger.info(
                f"[{self.name}] ✅ Completed in {self.metrics.execution_time_ms}ms "
                f"({self.metrics.llm_calls} LLM calls)"
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{self.name}] ❌ Failed: {error_msg}")
            logger.debug(traceback.format_exc())
            self.metrics.finish(status="failed", error=error_msg)
            output = self.get_fallback_output(error_msg)

        # --- Write metrics to state (reducer-compatible) ---
        # Only return THIS agent's metric — merge_dicts reducer merges all
        output["agent_metrics"] = {self.name: self.metrics.to_dict()}

        # Return a single-entry list — operator.add reducer concatenates
        output["execution_timeline"] = [{
            "agent": self.name,
            "status": self.metrics.status,
            "execution_time_ms": self.metrics.execution_time_ms,
            "timestamp": self.metrics.started_at,
        }]

        return output
