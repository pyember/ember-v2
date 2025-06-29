"""Model Context Protocol (MCP) integration for Ember.

This module provides MCP server implementations that expose Ember's capabilities
as tools, resources, and prompts accessible from any MCP-compatible client.
"""

from ember.integrations.mcp.server import EmberMCPServer

__all__ = ["EmberMCPServer"]
