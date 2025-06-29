"""Configuration migration utilities.

Migrates credentials and configuration from legacy formats to the
centralized context system. Idempotent and safe to run multiple times.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

from ember._internal.context import EmberContext
from ember.core.utils.logging import get_logger

logger = get_logger(__name__)


def _create_backup(file_path: Path) -> Path:
    """Create timestamped backup of file.

    Args:
        file_path: Path to file that needs backing up.

    Returns:
        Path to the created backup file with timestamp suffix.

    Raises:
        OSError: If backup creation fails due to permissions or disk space.

    Note:
        Backup filename format: original.ext.bak.YYYYMMDD_HHMMSS
        For files without extension: original.bak.YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # For files without extension, just add .bak.timestamp
    if file_path.suffix:
        backup = file_path.with_suffix(f"{file_path.suffix}.bak.{timestamp}")
    else:
        backup = file_path.with_suffix(f".bak.{timestamp}")
    shutil.copy2(file_path, backup)
    return backup


def migrate_credentials() -> bool:
    """Migrate credentials from legacy format to context system.

    Checks for credentials in the old location (~/.ember/credentials) and
    migrates them to the new context-aware system. Creates timestamped
    backups and uses a marker file to ensure idempotency.

    Returns:
        True if migration was performed, False if nothing to migrate or
        already migrated.

    Note:
        Migration is idempotent - safe to run multiple times.
        Original files are backed up, not deleted until migration succeeds.
    """
    old_credentials = Path.home() / ".ember" / "credentials"

    if not old_credentials.exists():
        logger.debug("No legacy credentials file found")
        return False

    try:
        with open(old_credentials) as f:
            credentials = json.load(f)

        if not credentials:
            logger.debug("Legacy credentials file is empty")
            return False

        # Check if already migrated
        migration_marker = old_credentials.parent / ".migration_complete"
        if migration_marker.exists():
            logger.info("Credentials already migrated")
            return False

        # Migrate to context system
        ctx = EmberContext.current()
        migrated_count = 0

        for provider, data in credentials.items():
            if isinstance(data, dict) and "api_key" in data:
                # Always migrate since we're moving from old to new format
                ctx.credential_manager.store(provider, data["api_key"])
                migrated_count += 1
                logger.info(f"Migrated credentials for {provider}")

        if migrated_count > 0:
            # Create backup and remove original
            backup = _create_backup(old_credentials)
            old_credentials.unlink()

            # Mark as migrated
            migration_marker.touch()

            print(f"Migrated {migrated_count} credential(s). Backup: {backup}")
            return True

        return False

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in credentials file: {e}")
        return False
    except OSError as e:
        logger.error(f"File system error during migration: {e}")
        return False


def migrate_config() -> bool:
    """Migrate config from old JSON format to YAML.

    Returns:
        True if migration performed, False if nothing to migrate
    """
    old_config = Path.home() / ".ember" / "config.json"

    if not old_config.exists():
        logger.debug("No legacy config file found")
        return False

    try:
        with open(old_config) as f:
            config = json.load(f)

        if not config:
            logger.debug("Legacy config file is empty")
            return False

        # Skip if new config already exists with content
        ctx = EmberContext.current()
        if ctx.get_config_path().exists() and ctx.get_all_config():
            logger.info("New config already exists, skipping migration")
            return False

        # Merge with existing config
        migrated_keys = []
        for key, value in config.items():
            if key not in ("version", "_migrated"):  # Skip metadata
                ctx.set_config(key, value)
                migrated_keys.append(key)
                logger.info(f"Migrated config: {key}")

        if migrated_keys:
            ctx.save()

            # Create backup and remove original
            backup = _create_backup(old_config)
            old_config.unlink()

            print(f"Migrated {len(migrated_keys)} config setting(s). Backup: {backup}")
            return True

        return False

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return False
    except OSError as e:
        logger.error(f"File system error during migration: {e}")
        return False


def main() -> Tuple[bool, bool]:
    """Run all migrations.

    Returns:
        Tuple of (credentials_migrated, config_migrated)
    """
    cred_migrated = migrate_credentials()
    config_migrated = migrate_config()

    if cred_migrated or config_migrated:
        logger.info("Migration completed successfully")
        print("Migration completed successfully.")
    else:
        logger.info("No migration needed")
        print("No migration needed.")

    return cred_migrated, config_migrated


if __name__ == "__main__":
    main()
