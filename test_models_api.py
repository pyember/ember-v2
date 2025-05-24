"""Test the models API to debug the issue."""

from ember.api import models

# Test 1: Check if models object has proper methods
print("Models API object:", models)
print("Has __call__:", hasattr(models, '__call__'))
print("Has instance:", hasattr(models, 'instance'))
print("Has list:", hasattr(models, 'list'))

# Test 2: Try listing models
try:
    available_models = models.list()
    print("\nAvailable models:", available_models)
except Exception as e:
    print("\nError listing models:", e)

# Test 3: Try calling models directly (will fail without API key)
try:
    # This should fail with ModelNotFoundError since no API keys are set
    response = models("openai:gpt-3.5-turbo", "Hello")
    print("\nResponse:", response)
except Exception as e:
    print("\nExpected error calling model:", type(e).__name__, str(e))

# Test 4: Check the actual type of models
print("\nType of models:", type(models))
print("Is ModelsAPI instance:", type(models).__name__)