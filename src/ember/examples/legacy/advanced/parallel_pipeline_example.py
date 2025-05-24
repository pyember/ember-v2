"""Parallel Pipeline Example with JIT.

This example demonstrates building a more complex pipeline with
branching paths and parallel execution using JIT-enabled operators.

Note: In the current implementation, each operator needs to be
decorated with @jit separately, and the graph must be built manually.
Future versions will simplify this process.

To run:
    uv run python src/ember/examples/advanced/parallel_pipeline_example.py
"""

import logging
import time
from typing import Any, Dict, Optional

# Use simplified imports
from ember.api.models import models
from ember.api.non import non
from ember.api.xcs import jit
# Import internals only when needed for advanced usage
from ember.xcs.engine.xcs_engine import (
    TopologicalSchedulerWithParallelDispatch,
    execute_graph,
)
from ember.xcs.graph.xcs_graph import XCSGraph

###############################################################################
# JIT-Decorated Ensemble Operators
###############################################################################


@jit()
class FactEnsemble(non.UniformEnsemble):
    """Ensemble focused on factual information."""

    pass


@jit()
class CreativeEnsemble(non.UniformEnsemble):
    """Ensemble focused on creative responses."""

    pass


@jit()
class DetailedEnsemble(non.UniformEnsemble):
    """Ensemble focused on detailed explanations."""

    pass


###############################################################################
# Pipeline Construction and Execution
###############################################################################
def build_multi_branch_pipeline(
    *, query: str, max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Build and execute a multi-branch pipeline with parallel execution.

    This creates a pipeline with three parallel branches (fact, creative, detailed),
    each with its own ensemble of models. The results are then synthesized by a judge.

    Args:
        query: The query to process
        max_workers: Maximum number of worker threads

    Returns:
        The final pipeline result
    """
    # Create the ensemble operators
    fact_ensemble = FactEnsemble(
        num_units=3,
        model_name="openai:gpt-4o-mini",
        temperature=0.3,  # Lower temperature for facts
    )

    creative_ensemble = CreativeEnsemble(
        num_units=3,
        model_name="openai:gpt-4o-mini",
        temperature=0.9,  # Higher temperature for creativity
    )

    detailed_ensemble = DetailedEnsemble(
        num_units=3,
        model_name="openai:gpt-4o-mini",
        temperature=0.5,  # Medium temperature
    )

    # Create the judge operator using non API
    judge = non.JudgeSynthesisOperator(
        model_name="openai:gpt-4o",
        temperature=0.0,
    )

    # Build the graph
    graph = XCSGraph()

    # Add nodes
    fact_id = graph.add_node(operator=fact_ensemble, node_id="fact_ensemble")
    creative_id = graph.add_node(
        operator=creative_ensemble, node_id="creative_ensemble"
    )
    detailed_id = graph.add_node(
        operator=detailed_ensemble, node_id="detailed_ensemble"
    )
    judge_id = graph.add_node(operator=judge, node_id="judge")

    # Connect all ensembles to the judge
    graph.add_edge(from_id=fact_id, to_id=judge_id)
    graph.add_edge(from_id=creative_id, to_id=judge_id)
    graph.add_edge(from_id=detailed_id, to_id=judge_id)

    # Use parallel execution
    workers = (
        max_workers or 12
    )  # Default to 12 workers (3 ensembles × 3 units + overhead)
    scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=workers)

    # Execute the graph
    start_time = time.perf_counter()
    result = execute_graph(
        graph=graph, global_input={"query": query}, scheduler=scheduler
    )
    end_time = time.perf_counter()

    logging.info(f"Pipeline execution completed in {end_time - start_time:.4f}s")

    return result


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of parallel pipeline with JIT-enabled operators."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Parallel Pipeline with JIT ===\n")

    # Example query that benefits from multiple perspectives
    query = "Explain the impact of artificial intelligence on society."

    print(f"Processing query: {query}")
    result = build_multi_branch_pipeline(query=query)

    print("\n=== Final Synthesized Answer ===")
    # Handle the actual output format from JudgeSynthesisOperator
    if "final_answer" in result:
        print(result["final_answer"])
    else:
        print("No final answer found in result")
        print(f"Available keys: {list(result.keys())}")

    # Print reasoning if available
    if "reasoning" in result:
        print("\n=== Judge's Reasoning ===")
        print(result["reasoning"])


if __name__ == "__main__":
    main()
