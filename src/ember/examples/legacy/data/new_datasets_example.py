"""Example for using specialized datasets (AIME, GPQA, Codeforces).

Run with: uvx python -m ember.examples.data.new_datasets_example [--skip-model-calls]
"""

import argparse
import logging
import sys
from typing import Dict

from ember.api import data, models
from ember.api.data import DataItem


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_aime(skip_model_calls: bool = False) -> bool:
    """Test AIME dataset loading and evaluation.

    Args:
        skip_model_calls: If True, skip LLM evaluation

    Returns:
        Success status
    """
    logger.info("Testing AIME dataset...")

    try:
        # Load dataset and ensure items are wrapped in DataItem for clean access
        aime_data = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data("aime", streaming=False)
        ]
        logger.info(f"✓ Loaded {len(aime_data)} AIME problems")

        if len(aime_data) == 0:
            logger.error("Dataset loaded but contains no problems")
            return False

        # Sample problem - clean attribute access
        problem = aime_data[0]
        logger.info(f"Sample problem: {problem.question[:100]}...")
        logger.info(f"Expected answer: {problem.answer}")

        # Test filtering
        aime_i = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("aime")
            .filter(lambda item: item.contest == 'I' if hasattr(item, 'contest') else False)
            .build()
        ]
        logger.info(f"✓ Filtered to {len(aime_i)} AIME-I problems")

        if not skip_model_calls:
            # Test evaluation
            logger.info("Testing model evaluation...")
            # Use the simplified models API
            response = models("gpt-4", problem.question)
            model_answer = response.text if hasattr(response, 'text') else str(response)

            # For AIME, we need numeric answer evaluation
            logger.info(f"Model answer: {model_answer}")
            logger.info(f"Expected: {problem.answer}")

        return True

    except Exception as e:
        logger.error(f"AIME dataset error: {e}")
        return False


def test_gpqa(skip_model_calls: bool = False) -> bool:
    """Test GPQA dataset loading and evaluation.

    Args:
        skip_model_calls: If True, skip LLM evaluation

    Returns:
        Success status
    """
    logger.info("Testing GPQA dataset...")

    try:
        # Load dataset with DataItem wrapper for clean access
        gpqa_data = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data("gpqa", streaming=False)
        ]
        logger.info(f"✓ Loaded {len(gpqa_data)} GPQA problems")

        if len(gpqa_data) == 0:
            logger.error("Dataset loaded but contains no questions")
            return False

        # Sample problem - clean attribute access
        problem = gpqa_data[0]
        logger.info(f"Sample question: {problem.question[:100]}...")
        
        # Clean access to options
        if problem.options:
            logger.info(f"Choices: {list(problem.options.keys())}")
        
        logger.info(f"Expected answer: {problem.answer}")

        # Format prompt for evaluation
        prompt = problem.question + "\n\n"
        if problem.options:
            for key, choice in problem.options.items():
                prompt += f"{key}. {choice}\n"

        if not skip_model_calls:
            # Test evaluation
            logger.info("Testing model evaluation...")
            # Use the simplified models API
            response = models("gpt-4", prompt)
            model_answer = response.text if hasattr(response, 'text') else str(response)

            # For multiple choice, extract the letter from response
            logger.info(f"Model answer: {model_answer}")
            logger.info(f"Expected: {problem.answer}")

        return True

    except Exception as e:
        if "gated dataset" in str(e).lower() or "authentication" in str(e).lower():
            logger.error("Authentication required for GPQA dataset")
            logger.info(
                "Request access at: https://huggingface.co/datasets/Idavidrein/gpqa"
            )
        else:
            logger.error(f"GPQA dataset error: {e}")
        return False


def test_codeforces(skip_model_calls: bool = False) -> bool:
    """Test Codeforces dataset loading and evaluation.

    Args:
        skip_model_calls: If True, skip LLM evaluation

    Returns:
        Success status
    """
    logger.info("Testing Codeforces dataset...")

    try:
        # Test with difficulty filtering
        cf_data = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("codeforces")
            .config(difficulty_range=(800, 1200))
            .sample(5)
            .build()
        ]
        logger.info(f"✓ Loaded {len(cf_data)} Codeforces problems")

        if len(cf_data) == 0:
            logger.error("Dataset loaded but contains no problems")
            return False

        # Sample problem - clean attribute access
        problem = cf_data[0]
        logger.info(f"Sample problem: {problem.question[:100]}...")
        
        # Access metadata cleanly through DataItem
        logger.info(f"Difficulty: {problem.difficulty if hasattr(problem, 'difficulty') else 'Unknown'}")
        logger.info(f"Tags: {problem.tags if hasattr(problem, 'tags') else 'None'}")

        # Skip model evaluation for code problems to avoid long execution
        if not skip_model_calls:
            logger.info("Skipping code evaluation to avoid long execution")

        return True

    except Exception as e:
        logger.error(f"Codeforces dataset error: {e}")
        return False


def summary(results: Dict[str, bool]) -> None:
    """Print summary of test results.

    Args:
        results: Dictionary of test results
    """
    logger.info("\n--- Dataset Test Summary ---")

    all_passed = True
    for dataset, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{dataset}: {status}")
        all_passed = all_passed and success

    if all_passed:
        logger.info("\nAll datasets loaded successfully!")
    else:
        logger.warning("\nSome datasets failed to load. See errors above.")

    # Show authentication hint if GPQA failed
    if not results.get("gpqa", True):
        logger.info("\nFor GPQA authentication:")
        logger.info("1. Run: huggingface-cli login")
        logger.info(
            "2. Request access: https://huggingface.co/datasets/Idavidrein/gpqa"
        )


def main() -> None:
    """Run the dataset test example."""
    parser = argparse.ArgumentParser(description="Test specialized datasets")
    parser.add_argument(
        "--skip-model-calls",
        action="store_true",
        help="Skip any calls to language models")
    args = parser.parse_args()

    results = {}

    results["aime"] = test_aime(args.skip_model_calls)
    results["gpqa"] = test_gpqa(args.skip_model_calls)
    results["codeforces"] = test_codeforces(args.skip_model_calls)

    summary(results)

    # Return non-zero exit code if any test failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()