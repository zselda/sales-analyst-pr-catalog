# Financial Intelligence Platform - Multi-Agent Swarm
# This package contains all LangGraph agents for the Hunter Swarm.
#
# Architecture:
#   BaseAgent → All agents inherit tracing, metrics, error isolation
#   Registry  → Agent capability metadata and dependency graph
#   Evaluation → Multi-dimensional quality assessment
