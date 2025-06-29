"""MCP server implementation for Ember."""

from typing import Any, Dict, List, Optional, Union
import json
import logging
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import (
        Tool,
        Resource,
        Prompt,
        TextContent,
        ImageContent,
        GetPromptResult,
        CallToolResult,
        ListResourcesResult,
        ReadResourceResult,
    )
except ImportError:
    raise ImportError(
        "MCP is required for this integration. Install with: pip install mcp"
    )

from ember.api import models, operators, data
from ember._internal.context.metrics import MetricsContext

logger = logging.getLogger(__name__)


@dataclass
class EmberToolResult:
    """Result from Ember tool execution."""

    content: str
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None


class EmberMCPServer:
    """MCP server exposing Ember capabilities.

    This server implements the Model Context Protocol to expose Ember's
    model orchestration capabilities as tools, resources, and prompts
    that can be accessed from any MCP-compatible client.

    Example:
        >>> from ember.integrations.mcp import EmberMCPServer
        >>>
        >>> # Create and run server
        >>> server = EmberMCPServer()
        >>> asyncio.run(server.run())
    """

    def __init__(self, name: str = "ember-mcp-server"):
        self.server = Server(name)
        self.metrics_context = MetricsContext()

        # Register capabilities
        self._register_tools()
        self._register_resources()
        self._register_prompts()

        logger.info(f"Initialized EmberMCPServer: {name}")

    def _register_tools(self):
        """Register Ember capabilities as MCP tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available Ember tools."""
            return [
                Tool(
                    name="ember_generate",
                    description="Generate text using any model in Ember's registry",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "model": {
                                "type": "string",
                                "description": "Model identifier (e.g., claude-3-opus)",
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (0.0-1.0)",
                                "default": 0.7,
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate",
                            },
                        },
                        "required": ["prompt", "model"],
                    },
                ),
                Tool(
                    name="ember_ensemble",
                    description="Run ensemble voting across multiple models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "models": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of model identifiers",
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["majority_vote", "weighted", "confidence"],
                                "description": "Voting strategy",
                                "default": "majority_vote",
                            },
                        },
                        "required": ["prompt", "models"],
                    },
                ),
                Tool(
                    name="ember_verify",
                    description="Verify and potentially improve model output",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Original prompt",
                            },
                            "output": {
                                "type": "string",
                                "description": "Output to verify",
                            },
                            "model": {
                                "type": "string",
                                "description": "Model to use for verification",
                            },
                            "criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Verification criteria",
                            },
                        },
                        "required": ["prompt", "output", "model"],
                    },
                ),
                Tool(
                    name="ember_compare_models",
                    description="Compare outputs from multiple models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "models": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Models to compare",
                            },
                            "include_metrics": {
                                "type": "boolean",
                                "description": "Include performance metrics",
                                "default": True,
                            },
                        },
                        "required": ["prompt", "models"],
                    },
                ),
                Tool(
                    name="ember_stream",
                    description="Stream text generation from a model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "model": {
                                "type": "string",
                                "description": "Model identifier",
                            },
                            "chunk_size": {
                                "type": "integer",
                                "description": "Approximate size of each chunk",
                                "default": 10,
                            },
                        },
                        "required": ["prompt", "model"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Optional[Dict[str, Any]] = None
        ) -> CallToolResult:
            """Execute an Ember tool."""
            if arguments is None:
                arguments = {}

            try:
                if name == "ember_generate":
                    result = await self._tool_generate(**arguments)
                elif name == "ember_ensemble":
                    result = await self._tool_ensemble(**arguments)
                elif name == "ember_verify":
                    result = await self._tool_verify(**arguments)
                elif name == "ember_compare_models":
                    result = await self._tool_compare_models(**arguments)
                elif name == "ember_stream":
                    result = await self._tool_stream(**arguments)
                else:
                    result = EmberToolResult(
                        content=f"Unknown tool: {name}", is_error=True
                    )

                return CallToolResult(
                    content=[TextContent(type="text", text=result.content)],
                    isError=result.is_error,
                )

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True,
                )

    async def _tool_generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> EmberToolResult:
        """Generate text using Ember."""
        try:
            ember_model = models.get_model(model)

            # Build parameters
            params = {"temperature": temperature}
            if max_tokens:
                params["max_tokens"] = max_tokens

            # Generate response
            with self.metrics_context.track():
                response = await asyncio.to_thread(
                    ember_model.generate, prompt, **params
                )

            # Get metrics
            metrics = self.metrics_context.get_last_metrics()

            return EmberToolResult(
                content=(
                    response.content if hasattr(response, "content") else str(response)
                ),
                metadata={
                    "model": model,
                    "usage": metrics.get("usage", {}),
                    "latency_ms": metrics.get("latency_ms", 0),
                },
            )

        except Exception as e:
            return EmberToolResult(
                content=f"Generation failed: {str(e)}", is_error=True
            )

    async def _tool_ensemble(
        self, prompt: str, models: List[str], strategy: str = "majority_vote"
    ) -> EmberToolResult:
        """Run ensemble across multiple models."""
        try:
            # Create operators for each model
            ops = [operators.Operator(model=m) for m in models]

            # Create ensemble
            ensemble = operators.EnsembleOperator(operators=ops, strategy=strategy)

            # Run ensemble
            result = await asyncio.to_thread(ensemble.run, prompt)

            # Format response
            response_data = {
                "consensus": result.content,
                "votes": result.metadata.get("votes", {}),
                "confidence": result.metadata.get("confidence", 0.0),
                "models_used": models,
            }

            return EmberToolResult(
                content=json.dumps(response_data, indent=2), metadata=response_data
            )

        except Exception as e:
            return EmberToolResult(content=f"Ensemble failed: {str(e)}", is_error=True)

    async def _tool_verify(
        self, prompt: str, output: str, model: str, criteria: Optional[List[str]] = None
    ) -> EmberToolResult:
        """Verify and improve output."""
        try:
            verifier = operators.VerifierOperator(model=model, criteria=criteria or [])

            result = await asyncio.to_thread(verifier.verify, prompt, output)

            response_data = {
                "is_valid": result.is_valid,
                "issues": result.issues,
                "improved_output": (
                    result.improved_output if not result.is_valid else output
                ),
            }

            return EmberToolResult(
                content=json.dumps(response_data, indent=2), metadata=response_data
            )

        except Exception as e:
            return EmberToolResult(
                content=f"Verification failed: {str(e)}", is_error=True
            )

    async def _tool_compare_models(
        self, prompt: str, models: List[str], include_metrics: bool = True
    ) -> EmberToolResult:
        """Compare outputs from multiple models."""
        try:
            results = {}

            for model_name in models:
                ember_model = models.get_model(model_name)

                with self.metrics_context.track():
                    response = await asyncio.to_thread(ember_model.generate, prompt)

                metrics = self.metrics_context.get_last_metrics()

                results[model_name] = {
                    "output": (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    ),
                    "metrics": metrics if include_metrics else {},
                }

            return EmberToolResult(
                content=json.dumps(results, indent=2), metadata={"comparison": results}
            )

        except Exception as e:
            return EmberToolResult(
                content=f"Comparison failed: {str(e)}", is_error=True
            )

    async def _tool_stream(
        self, prompt: str, model: str, chunk_size: int = 10
    ) -> EmberToolResult:
        """Stream text generation."""
        try:
            ember_model = models.get_model(model)
            chunks = []

            # Collect chunks (in real implementation, this would stream)
            async for chunk in ember_model.astream(prompt):
                chunks.append(chunk)

            # Format as streaming response
            response_data = {
                "model": model,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "complete_text": "".join(chunks),
            }

            return EmberToolResult(
                content=json.dumps(response_data, indent=2), metadata=response_data
            )

        except Exception as e:
            return EmberToolResult(content=f"Streaming failed: {str(e)}", is_error=True)

    def _register_resources(self):
        """Register Ember data as MCP resources."""

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available Ember resources."""
            return [
                Resource(
                    uri="ember://models/registry",
                    name="Model Registry",
                    description="Complete list of available models with capabilities and pricing",
                    mimeType="application/json",
                ),
                Resource(
                    uri="ember://models/costs",
                    name="Model Costs",
                    description="Current pricing information for all models",
                    mimeType="application/json",
                ),
                Resource(
                    uri="ember://metrics/usage",
                    name="Usage Metrics",
                    description="Current session usage statistics",
                    mimeType="application/json",
                ),
                Resource(
                    uri="ember://operators/types",
                    name="Operator Types",
                    description="Available operator types and their descriptions",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> ReadResourceResult:
            """Read an Ember resource."""
            try:
                if uri == "ember://models/registry":
                    content = await self._get_model_registry()
                elif uri == "ember://models/costs":
                    content = await self._get_model_costs()
                elif uri == "ember://metrics/usage":
                    content = await self._get_usage_metrics()
                elif uri == "ember://operators/types":
                    content = await self._get_operator_types()
                else:
                    content = f"Unknown resource: {uri}"

                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text", text=content, mimeType="application/json"
                        )
                    ]
                )

            except Exception as e:
                logger.error(f"Resource read failed: {e}")
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps({"error": str(e)}),
                            mimeType="application/json",
                        )
                    ]
                )

    async def _get_model_registry(self) -> str:
        """Get model registry information."""
        registry = models.get_registry()

        formatted_registry = {}
        for model_id, info in registry.items():
            formatted_registry[model_id] = {
                "provider": info.provider,
                "capabilities": info.capabilities,
                "context_length": info.context_length,
                "supports_streaming": info.supports_streaming,
                "supports_tools": info.supports_tools,
            }

        return json.dumps(formatted_registry, indent=2)

    async def _get_model_costs(self) -> str:
        """Get model cost information."""
        registry = models.get_registry()

        costs = {}
        for model_id, info in registry.items():
            costs[model_id] = {
                "input_cost_per_1k": info.input_cost,
                "output_cost_per_1k": info.output_cost,
                "currency": "USD",
            }

        return json.dumps(costs, indent=2)

    async def _get_usage_metrics(self) -> str:
        """Get current usage metrics."""
        # This would aggregate real metrics in production
        metrics = {
            "session_start": datetime.now().isoformat(),
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "models_used": [],
            "average_latency_ms": 0,
        }

        return json.dumps(metrics, indent=2)

    async def _get_operator_types(self) -> str:
        """Get available operator types."""
        operator_info = {
            "Operator": {
                "description": "Base operator for single model calls",
                "use_case": "Simple text generation tasks",
            },
            "EnsembleOperator": {
                "description": "Combines multiple models with voting strategies",
                "use_case": "High-stakes decisions requiring consensus",
            },
            "VerifierOperator": {
                "description": "Verifies and improves model outputs",
                "use_case": "Quality assurance and output validation",
            },
            "JudgeSynthesisOperator": {
                "description": "Judges and synthesizes multiple outputs",
                "use_case": "Complex reasoning and synthesis tasks",
            },
        }

        return json.dumps(operator_info, indent=2)

    def _register_prompts(self):
        """Register Ember prompt templates."""

        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name="code_review",
                    description="Comprehensive code review with multiple perspectives",
                    arguments=[
                        {
                            "name": "code",
                            "description": "Code to review",
                            "required": True,
                        },
                        {
                            "name": "language",
                            "description": "Programming language",
                            "required": False,
                        },
                    ],
                ),
                Prompt(
                    name="chain_of_thought",
                    description="Step-by-step reasoning for complex problems",
                    arguments=[
                        {
                            "name": "problem",
                            "description": "Problem to solve",
                            "required": True,
                        }
                    ],
                ),
                Prompt(
                    name="compare_approaches",
                    description="Compare multiple approaches to a problem",
                    arguments=[
                        {
                            "name": "problem",
                            "description": "Problem description",
                            "required": True,
                        },
                        {
                            "name": "approaches",
                            "description": "List of approaches",
                            "required": True,
                        },
                    ],
                ),
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: Optional[Dict[str, str]] = None
        ) -> GetPromptResult:
            """Get a specific prompt template."""
            if arguments is None:
                arguments = {}

            try:
                if name == "code_review":
                    messages = self._prompt_code_review(**arguments)
                elif name == "chain_of_thought":
                    messages = self._prompt_chain_of_thought(**arguments)
                elif name == "compare_approaches":
                    messages = self._prompt_compare_approaches(**arguments)
                else:
                    messages = [{"role": "user", "content": f"Unknown prompt: {name}"}]

                return GetPromptResult(
                    description=f"Prompt template: {name}", messages=messages
                )

            except Exception as e:
                logger.error(f"Prompt generation failed: {e}")
                return GetPromptResult(
                    description="Error generating prompt",
                    messages=[{"role": "user", "content": f"Error: {str(e)}"}],
                )

    def _prompt_code_review(
        self, code: str, language: str = "python"
    ) -> List[Dict[str, str]]:
        """Generate code review prompt."""
        return [
            {
                "role": "system",
                "content": f"You are an expert {language} code reviewer. Review code for:\n"
                "1. Correctness and logic errors\n"
                "2. Performance and efficiency\n"
                "3. Security vulnerabilities\n"
                "4. Code style and best practices\n"
                "5. Maintainability and documentation",
            },
            {
                "role": "user",
                "content": f"Please review this {language} code:\n\n```{language}\n{code}\n```",
            },
        ]

    def _prompt_chain_of_thought(self, problem: str) -> List[Dict[str, str]]:
        """Generate chain of thought prompt."""
        return [
            {
                "role": "system",
                "content": "You are a logical reasoning expert. Break down problems step by step, "
                "showing your work at each stage. Be thorough and explicit in your reasoning.",
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\n"
                "Please solve this step by step:\n"
                "1. First, identify what we know\n"
                "2. Then, determine what we need to find\n"
                "3. Next, outline the approach\n"
                "4. Finally, work through the solution\n"
                "5. Verify the answer",
            },
        ]

    def _prompt_compare_approaches(
        self, problem: str, approaches: str
    ) -> List[Dict[str, str]]:
        """Generate comparison prompt."""
        return [
            {
                "role": "system",
                "content": "You are an analytical expert who excels at comparing different approaches. "
                "Provide balanced, objective analysis considering multiple factors.",
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\n"
                f"Approaches to compare:\n{approaches}\n\n"
                "Please analyze each approach considering:\n"
                "- Effectiveness\n"
                "- Efficiency\n"
                "- Cost\n"
                "- Complexity\n"
                "- Risks\n"
                "- Long-term implications\n\n"
                "Provide a recommendation with rationale.",
            },
        ]

    async def run(self, transport: str = "stdio"):
        """Run the MCP server.

        Args:
            transport: Transport method ("stdio" or "http")
        """
        logger.info(f"Starting EmberMCPServer with {transport} transport")

        if transport == "stdio":
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="ember-mcp", server_version="1.0.0"
                    ),
                )
        elif transport == "http":
            # HTTP transport would be implemented here
            raise NotImplementedError("HTTP transport not yet implemented")
        else:
            raise ValueError(f"Unknown transport: {transport}")
