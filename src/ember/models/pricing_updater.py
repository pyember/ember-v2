"""Automated pricing updater using LLMs to fetch and parse pricing data.

This module implements a self-maintaining pricing system that uses our own
models API to programmatically fetch and update pricing information.

Following Jeff Dean's philosophy: "Build systems that maintain themselves."
"""

import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from ember.api import models

logger = logging.getLogger(__name__)


class PricingUpdater:
    """Automated pricing updater using LLMs."""
    
    def __init__(self, pricing_yaml_path: Optional[Path] = None):
        self.pricing_path = pricing_yaml_path or Path(__file__).parent / "pricing.yaml"
        self.sources = {
            "openai": "https://openai.com/api/pricing/",
            "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models/overview#model-pricing",
            "google": "https://ai.google.dev/gemini-api/docs/pricing"
        }
    
    def fetch_provider_pricing(self, provider: str, url: str) -> Dict[str, Dict]:
        """Use Claude to fetch and parse pricing for a provider."""
        prompt = f"""Please visit {url} and extract the current API pricing information.
        
For each model, provide:
1. Model name/ID exactly as shown
2. Input price per million tokens in USD
3. Output price per million tokens in USD
4. Context window size (if available)

Format your response as YAML like this:
```yaml
model-id-1:
  input: 15.0   # $15 per 1M tokens
  output: 75.0  # $75 per 1M tokens
  context: 200000
model-id-2:
  input: 3.0
  output: 15.0
  context: 200000
```

Only include models that have API pricing (not chat/consumer pricing).
Be precise with model IDs - use exact names from the pricing page."""
        
        try:
            # Use Claude Opus for best accuracy
            response = models("claude-3-opus", prompt)
            
            # Extract YAML from response
            text = response.text
            yaml_start = text.find("```yaml")
            yaml_end = text.find("```", yaml_start + 7)
            
            if yaml_start != -1 and yaml_end != -1:
                yaml_content = text[yaml_start + 7:yaml_end].strip()
                return yaml.safe_load(yaml_content)
            else:
                logger.warning(f"No YAML found in response for {provider}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to fetch pricing for {provider}: {e}")
            return {}
    
    def update_pricing(self, dry_run: bool = True) -> Dict:
        """Update pricing YAML with latest data."""
        logger.info("Starting automated pricing update...")
        
        # Load existing pricing
        if self.pricing_path.exists():
            with open(self.pricing_path) as f:
                current_data = yaml.safe_load(f)
        else:
            current_data = {
                "version": "1.0",
                "providers": {}
            }
        
        # Fetch new pricing for each provider
        new_data = {
            "version": current_data.get("version", "1.0"),
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "providers": {}
        }
        
        for provider, url in self.sources.items():
            logger.info(f"Fetching {provider} pricing from {url}...")
            
            models_data = self.fetch_provider_pricing(provider, url)
            
            if models_data:
                new_data["providers"][provider] = {
                    "source": url,
                    "models": models_data
                }
                logger.info(f"Found {len(models_data)} models for {provider}")
            else:
                # Keep existing data if fetch failed
                if provider in current_data.get("providers", {}):
                    new_data["providers"][provider] = current_data["providers"][provider]
                    logger.warning(f"Keeping existing data for {provider}")
        
        # Compare and report changes
        changes = self._compare_pricing(current_data, new_data)
        
        if changes:
            logger.info(f"Found {len(changes)} pricing changes:")
            for change in changes:
                logger.info(f"  - {change}")
        else:
            logger.info("No pricing changes detected")
        
        # Save if not dry run
        if not dry_run and changes:
            # Backup current file
            if self.pricing_path.exists():
                backup_path = self.pricing_path.with_suffix(".yaml.bak")
                self.pricing_path.rename(backup_path)
                logger.info(f"Backed up to {backup_path}")
            
            # Write new data
            with open(self.pricing_path, 'w') as f:
                yaml.dump(new_data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Updated {self.pricing_path}")
        
        return new_data
    
    def _compare_pricing(self, old_data: Dict, new_data: Dict) -> List[str]:
        """Compare old and new pricing to find changes."""
        changes = []
        
        old_providers = old_data.get("providers", {})
        new_providers = new_data.get("providers", {})
        
        # Check each provider
        for provider in set(old_providers.keys()) | set(new_providers.keys()):
            old_models = old_providers.get(provider, {}).get("models", {})
            new_models = new_providers.get(provider, {}).get("models", {})
            
            # Check each model
            for model in set(old_models.keys()) | set(new_models.keys()):
                if model not in old_models:
                    changes.append(f"{provider}/{model}: NEW MODEL")
                elif model not in new_models:
                    changes.append(f"{provider}/{model}: REMOVED")
                else:
                    old_m = old_models[model]
                    new_m = new_models[model]
                    
                    if old_m.get("input") != new_m.get("input"):
                        changes.append(
                            f"{provider}/{model}: input ${old_m.get('input')} → ${new_m.get('input')}"
                        )
                    if old_m.get("output") != new_m.get("output"):
                        changes.append(
                            f"{provider}/{model}: output ${old_m.get('output')} → ${new_m.get('output')}"
                        )
        
        return changes
    
    def validate_pricing(self, data: Dict) -> List[str]:
        """Validate pricing data structure."""
        errors = []
        
        if "providers" not in data:
            errors.append("Missing 'providers' key")
            return errors
        
        for provider, provider_data in data["providers"].items():
            if "models" not in provider_data:
                errors.append(f"{provider}: Missing 'models' key")
                continue
            
            for model, pricing in provider_data["models"].items():
                if "input" not in pricing:
                    errors.append(f"{provider}/{model}: Missing 'input' price")
                elif not isinstance(pricing["input"], (int, float)) or pricing["input"] < 0:
                    errors.append(f"{provider}/{model}: Invalid 'input' price")
                
                if "output" not in pricing:
                    errors.append(f"{provider}/{model}: Missing 'output' price")
                elif not isinstance(pricing["output"], (int, float)) or pricing["output"] < 0:
                    errors.append(f"{provider}/{model}: Invalid 'output' price")
                
                if "context" in pricing:
                    if not isinstance(pricing["context"], int) or pricing["context"] < 1:
                        errors.append(f"{provider}/{model}: Invalid 'context' size")
        
        return errors


def update_pricing_cli():
    """CLI entry point for updating pricing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update model pricing data")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without updating file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing pricing.yaml"
    )
    
    args = parser.parse_args()
    
    updater = PricingUpdater()
    
    if args.validate_only:
        # Validate existing file
        with open(updater.pricing_path) as f:
            data = yaml.safe_load(f)
        
        errors = updater.validate_pricing(data)
        if errors:
            print(f"Found {len(errors)} validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("✓ Pricing data is valid")
            return 0
    else:
        # Update pricing
        try:
            updater.update_pricing(dry_run=args.dry_run)
            return 0
        except Exception as e:
            print(f"Error updating pricing: {e}")
            return 1


if __name__ == "__main__":
    exit(update_pricing_cli())