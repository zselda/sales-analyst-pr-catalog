"""
Agent Registry
================
Central registry for all agents with capability metadata,
dependency tracking, and parallelization detection.
"""
 
from typing import Optional
import logging
 
logger = logging.getLogger("swarm.registry")
 
 
class AgentCapability:
    """Metadata describing an agent's capabilities and dependencies."""
 
    def __init__(
        self,
        name: str,
        description: str,
        domain: str,
        required_inputs: list[str],
        output_keys: list[str],
        depends_on: list[str] | None = None,
        parallel_group: str | None = None,
        timeout_seconds: int = 60,
    ):
        self.name = name
        self.description = description
        self.domain = domain
        self.required_inputs = required_inputs
        self.output_keys = output_keys
        self.depends_on = depends_on or []
        self.parallel_group = parallel_group
        self.timeout_seconds = timeout_seconds
 
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "required_inputs": self.required_inputs,
            "output_keys": self.output_keys,
            "depends_on": self.depends_on,
            "parallel_group": self.parallel_group,
            "timeout_seconds": self.timeout_seconds,
        }
 
 
class AgentRegistry:
    """
    Central registry for all swarm agents.
 
    Provides:
    - Agent registration and lookup
    - Dependency graph for automatic parallelization detection
    - Health/readiness status tracking
    """
 
    def __init__(self):
        self._agents: dict[str, AgentCapability] = {}
        self._health: dict[str, str] = {}
 
    def register(self, capability: AgentCapability) -> None:
        """Register an agent with its capabilities."""
        self._agents[capability.name] = capability
        self._health[capability.name] = "ready"
        logger.info(f"[Registry] Registered agent: {capability.name} ({capability.domain})")
 
    def get(self, name: str) -> Optional[AgentCapability]:
        """Get agent capability by name."""
        return self._agents.get(name)
 
    def get_parallel_groups(self) -> dict[str, list[str]]:
        """
        Identify agents that can run in parallel.
        Returns groups of agent names that share a parallel_group.
        """
        groups: dict[str, list[str]] = {}
        for name, cap in self._agents.items():
            if cap.parallel_group:
                groups.setdefault(cap.parallel_group, []).append(name)
        return {k: v for k, v in groups.items() if len(v) > 1}
 
    def get_dependency_order(self) -> list[list[str]]:
        """
        Return agents grouped by execution order.
        Agents with no unmet dependencies can run in parallel.
        """
        resolved: set[str] = set()
        order: list[list[str]] = []
        remaining = set(self._agents.keys())
 
        while remaining:
            batch = []
            for name in list(remaining):
                cap = self._agents[name]
                if all(dep in resolved for dep in cap.depends_on):
                    batch.append(name)
            if not batch:
                # Circular dependency or unresolvable — add remaining as single batch
                order.append(list(remaining))
                break
            order.append(batch)
            resolved.update(batch)
            remaining -= set(batch)
 
        return order
 
    def set_health(self, name: str, status: str) -> None:
        """Update agent health status."""
        self._health[name] = status
 
    def get_all_status(self) -> dict[str, dict]:
        """Get status of all registered agents."""
        return {
            name: {
                **cap.to_dict(),
                "health": self._health.get(name, "unknown"),
            }
            for name, cap in self._agents.items()
        }
 
    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
 
 
# ============================================================================
# Global registry instance — populated during agent module imports
# ============================================================================
registry = AgentRegistry()
 
# Register all agents with their capabilities
registry.register(AgentCapability(
    name="data_ingestion",
    description="Standardize Mizan data with Turkish Chart of Accounts classification",
    domain="data_processing",
    required_inputs=["mizan_data"],
    output_keys=["standardized_mizan"],
    depends_on=[],
    timeout_seconds=10,
))
 
registry.register(AgentCapability(
    name="quant_analyst",
    description="Calculate financial ratios from standardized Mizan data",
    domain="financial_analysis",
    required_inputs=["standardized_mizan"],
    output_keys=["financial_ratios"],
    depends_on=["data_ingestion"],
    timeout_seconds=30,
))
 
registry.register(AgentCapability(
    name="verifier",
    description="Validate financial ratios against Turkish Chart of Accounts rules",
    domain="audit",
    required_inputs=["financial_ratios"],
    output_keys=["verification_status", "verification_errors"],
    depends_on=["quant_analyst"],
    timeout_seconds=30,
))
 
registry.register(AgentCapability(
    name="behavioral",
    description="Analyze transaction patterns, cash flow, and competitor activity",
    domain="behavioral_analysis",
    required_inputs=["transactions_data"],
    output_keys=["behavioral_insights"],
    depends_on=["verifier"],
    parallel_group="post_verification",
    timeout_seconds=45,
))
 
registry.register(AgentCapability(
    name="network_mapper",
    description="Build commercial network graph from Mizan sub-accounts",
    domain="network_analysis",
    required_inputs=["standardized_mizan"],
    output_keys=["network_data"],
    depends_on=["verifier"],
    parallel_group="post_verification",
    timeout_seconds=15,
))
 
registry.register(AgentCapability(
    name="strategist",
    description="Generate data-driven sales strategy report",
    domain="strategy",
    required_inputs=["financial_ratios", "behavioral_insights", "network_data"],
    output_keys=["strategy_report"],
    depends_on=["behavioral", "network_mapper"],
    timeout_seconds=60,
))
