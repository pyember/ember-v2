"""Tests for EmberContext system - improved version.

Principles:
- Hermetic tests with tmp_ctx fixture
- Parametrized tests for comprehensive coverage
- Fast, deterministic, parallelizable
"""

import asyncio
import threading
import time

import pytest
import yaml

from ember.context import (
    EmberContext,
    context,
)


class TestContextCore:
    """Core context functionality."""

    def test_singleton(self, tmp_ctx):
        """Singleton returns same instance."""
        assert EmberContext.current() is EmberContext.current()

    def test_isolation(self, tmp_ctx):
        """Isolated contexts are independent."""
        # tmp_ctx is already isolated from global
        isolated = EmberContext(isolated=True)

        # Changes to isolated don't affect tmp_ctx
        isolated.set_config("test", "isolated")
        assert tmp_ctx.get_config("test") is None

    @pytest.mark.parametrize(
        "key,value",
        [
            ("simple", "value"),
            ("nested.key", "value"),
            ("deep.nested.key", "value"),
            ("", None),  # Edge case
            (None, None),  # Edge case
        ],
    )
    def test_config_operations(self, tmp_ctx, key, value):
        """Config get/set with various inputs."""
        if key:
            tmp_ctx.set_config(key, value)
            assert tmp_ctx.get_config(key) == value

            # Verify structure is created
            if "." in key:
                parts = key.split(".")
                parent = tmp_ctx.get_config(parts[0])
                assert isinstance(parent, dict)
        else:
            # Edge cases
            assert tmp_ctx.get_config(key) is None


class TestCredentialPrecedence:
    """Credential precedence testing."""

    @pytest.mark.parametrize(
        "env_val,file_val,config_val,expected",
        [
            ("env-key", "file-key", "config-key", "env-key"),
            (None, "file-key", "config-key", "file-key"),
            (None, None, "config-key", "config-key"),
            (None, None, None, None),
        ],
    )
    def test_precedence_matrix(self, tmp_ctx, monkeypatch, env_val, file_val, config_val, expected):
        """Test credential precedence: env > file > config."""
        provider = "test"
        env_var = "TEST_API_KEY"

        # Setup environment
        if env_val:
            monkeypatch.setenv(env_var, env_val)
        else:
            monkeypatch.delenv(env_var, raising=False)

        # Setup file credential
        if file_val:
            tmp_ctx.credential_manager.save_api_key(provider, file_val)

        # Setup config
        if config_val:
            tmp_ctx.set_config(f"providers.{provider}.api_key", config_val)

        # Test precedence
        assert tmp_ctx.get_credential(provider, env_var) == expected


class TestThreadSafety:
    """Thread-safe context operations."""

    def test_concurrent_isolation(self, tmp_ctx):
        """Concurrent contexts remain isolated."""
        results = {}
        errors = []
        barrier = threading.Barrier(10)

        def worker(worker_id):
            try:
                # Create isolated context
                with context.manager(worker_id=worker_id):
                    barrier.wait()  # Synchronize start

                    # Each worker sets and reads its own value
                    ctx = context.get()
                    ctx.set_config(f"worker.{worker_id}", worker_id)

                    # Verify isolation
                    for i in range(10):
                        if i != worker_id:
                            assert ctx.get_config(f"worker.{i}") is None

                    results[worker_id] = ctx.get_config(f"worker.{worker_id}")
            except Exception as e:
                errors.append((worker_id, e))

        # Run workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify results
        assert not errors
        assert results == {i: i for i in range(10)}


class TestAsyncPropagation:
    """Async context propagation."""

    @pytest.mark.asyncio
    async def test_context_across_await(self, tmp_ctx):
        """Context persists across await boundaries."""

        async def async_operation():
            ctx1 = context.get()
            ctx1.set_config("async_test", "before")

            await asyncio.sleep(0.001)

            ctx2 = context.get()
            assert ctx1 is ctx2
            assert ctx2.get_config("async_test") == "before"

            return True

        with context.manager():
            result = await async_operation()
            assert result

    @pytest.mark.asyncio
    async def test_concurrent_async_isolation(self, tmp_ctx):
        """Concurrent async tasks have isolated contexts."""
        results = []

        async def task(task_id, model):
            with context.manager(model=model):
                await asyncio.sleep(0.001)
                ctx = context.get()
                results.append((task_id, ctx.get_config("model")))

        # Run concurrent tasks
        await asyncio.gather(
            task(1, "gpt-4"),
            task(2, "claude-3"),
            task(3, "gemini"),
        )

        # Each task saw its own context
        assert sorted(results) == [(1, "gpt-4"), (2, "claude-3"), (3, "gemini")]


class TestErrorHandling:
    """Error handling and edge cases."""

    def test_corrupted_yaml(self, tmp_ctx, tmp_path):
        """Corrupted YAML files don't crash."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{ this: is: : invalid }")

        ctx = EmberContext(isolated=True)
        ctx._config_file = config_file

        # Should not raise
        ctx.reload()

        # Should have some config (YAML is lenient)
        assert isinstance(ctx.get_all_config(), dict)

    def test_missing_directory(self, tmp_ctx, tmp_path):
        """Missing directories are created."""
        config_path = tmp_path / "new" / "dir" / "config.yaml"

        ctx = EmberContext(isolated=True)
        ctx._config_file = config_path
        ctx.set_config("test", "value")

        # Save should create directories
        ctx.save()

        assert config_path.exists()
        assert config_path.parent.is_dir()

    @pytest.mark.parametrize(
        "bad_input",
        [
            "",  # Empty string
            None,  # None
            123,  # Wrong type
            "no.such.key.exists.anywhere",  # Deep missing key
        ],
    )
    def test_get_config_edge_cases(self, tmp_ctx, bad_input):
        """get_config handles edge cases gracefully."""
        result = tmp_ctx.get_config(bad_input)
        assert result is None  # Should return None, not crash


class TestPersistence:
    """Configuration persistence."""

    def test_atomic_save(self, tmp_ctx, tmp_path):
        """Saves are atomic."""
        # Set up context with known path
        config_file = tmp_path / "ember" / "config.yaml"
        ctx = EmberContext(isolated=True)
        ctx._config_file = config_file

        # Save data
        ctx.set_config("test.data", {"key": "value"})
        ctx.save()

        # Verify file exists with correct content
        assert config_file.exists()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["test"]["data"]["key"] == "value"

        # Verify no temp files left behind
        temp_files = list(config_file.parent.glob("*.tmp"))
        assert not temp_files


class TestPerformance:
    """Performance characteristics."""

    def test_context_creation_speed(self, tmp_ctx):
        """Context creation is fast."""
        # Warm up
        EmberContext(isolated=True)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            EmberContext(isolated=True)
        elapsed = time.perf_counter() - start

        # Should be fast (< 50ms for 100 contexts)
        assert elapsed < 0.05

    def test_config_lookup_performance(self, tmp_ctx):
        """Deep config lookups are fast."""
        # Build nested config
        for i in range(50):
            tmp_ctx.set_config(f"level1.level2.level3.level4.item{i}", i)

        # Measure lookups
        start = time.perf_counter()
        for _ in range(1000):
            for i in range(50):
                value = tmp_ctx.get_config(f"level1.level2.level3.level4.item{i}")
                assert value == i
        elapsed = time.perf_counter() - start

        # Should be fast (< 100ms for 50k lookups)
        assert elapsed < 0.1


@pytest.mark.parametrize("permission", [0o600, 0o644, 0o666])
def test_credential_file_permissions(tmp_ctx, permission):
    """Credential files have correct permissions."""
    # Save a credential
    tmp_ctx.credential_manager.save_api_key("test", "secret")

    # Check file permissions
    cred_file = tmp_ctx.credential_manager.credentials_file
    assert cred_file.exists()

    # Should always be 0o600 regardless of umask
    actual_mode = cred_file.stat().st_mode & 0o777
    assert actual_mode == 0o600
