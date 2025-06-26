"""Your first Ember program!"""
from ember.api import models, Models

# Direct API call
response = models("claude-3-haiku", "Hello! Tell me something interesting.")
print(response)

# Using model constants for autocomplete
response = models(Models.CLAUDE_3_HAIKU, "What can you help me with?")
print(response)

# Create a reusable assistant
assistant = models.instance("claude-3-haiku", temperature=0.7)
print(assistant("Tell me a joke"))
print(assistant("Now explain why it's funny"))
