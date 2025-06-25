#!/usr/bin/env python3
"""Test CLI integration with context system."""

import subprocess
import sys
import os
from pathlib import Path

# Get config path from public API
sys.path.insert(0, str(Path(__file__).parent / "src"))
from ember._internal.context import EmberContext

test_config = EmberContext.get_config_path()
if test_config.exists():
    print(f"Backing up existing config to {test_config}.bak")
    test_config.rename(test_config.with_suffix(".yaml.bak"))

def run_ember(*args):
    """Run ember CLI command."""
    cmd = [sys.executable, "-m", "ember.cli.main"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

print("Testing CLI Context Integration")
print("=" * 50)

# Test 1: Configure set
print("\n1. Testing 'ember configure set'...")
result = run_ember("configure", "set", "test.value", "42")
if result.returncode == 0:
    print("✓ Set test.value = 42")
else:
    print(f"✗ Failed: {result.stderr}")

# Test 2: Configure get
print("\n2. Testing 'ember configure get'...")
result = run_ember("configure", "get", "test.value")
if result.returncode == 0 and "42" in result.stdout:
    print("✓ Retrieved test.value = 42")
else:
    print(f"✗ Failed: {result.stdout} {result.stderr}")

# Test 3: Configure list
print("\n3. Testing 'ember configure list'...")
result = run_ember("configure", "list", "--format", "json")
if result.returncode == 0:
    print("✓ Listed configuration")
    import json
    try:
        config = json.loads(result.stdout)
        print(f"  Config has {len(config)} top-level keys")
    except:
        pass
else:
    print(f"✗ Failed: {result.stderr}")

# Test 4: Set default model
print("\n4. Setting default model...")
result = run_ember("configure", "set", "models.default", "gpt-3.5-turbo")
if result.returncode == 0:
    print("✓ Set default model")
else:
    print(f"✗ Failed: {result.stderr}")

# Test 5: Test command uses default
print("\n5. Testing 'ember test' with default model...")
# This will fail without API key, but we're checking it uses config
result = run_ember("test")
if "gpt-3.5-turbo" in result.stdout or "gpt-3.5-turbo" in result.stderr:
    print("✓ Test command uses configured default model")
else:
    print(f"✗ Test command not using default: {result.stdout} {result.stderr}")

# Test 6: Models command
print("\n6. Testing 'ember models'...")
result = run_ember("models")
if result.returncode == 0 and "gpt-4" in result.stdout:
    print("✓ Models command works")
else:
    print(f"✗ Failed: {result.stderr}")

# Test 7: Persistence
print("\n7. Testing configuration persistence...")
# The config should persist, so get the value again
result = run_ember("configure", "get", "test.value")
if result.returncode == 0 and "42" in result.stdout:
    print("✓ Configuration persisted across invocations")
else:
    print(f"✗ Configuration not persisted: {result.stdout}")

# Test 8: Check config file exists
if test_config.exists():
    print(f"\n✓ Config file created at {test_config}")
    with open(test_config) as f:
        print("Contents preview:")
        content = f.read()
        print(content[:200] + "..." if len(content) > 200 else content)
else:
    print(f"\n✗ Config file not found at {test_config}")

print("\n" + "=" * 50)
print("CLI integration test complete!")

# Restore original config if it existed
backup = test_config.with_suffix(".yaml.bak")
if backup.exists():
    print(f"\nRestoring original config from {backup}")
    if test_config.exists():
        test_config.unlink()
    backup.rename(test_config)