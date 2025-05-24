"""Tool for exploring Ember's specialized datasets (AIME, GPQA, Codeforces).

This utility helps users explore sample problems from each specialized dataset,
understand their structure, and test their integration with the Ember framework.

Usage:
    python -m ember.examples.data.explore_datasets --dataset aime --count 3
"""

import argparse
import logging

from ember.api import data
from ember.api.data import DataItem

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def explore_aime(count: int = 3) -> None:
    """Explore AIME dataset structure.

    Args:
        count: Number of examples to show
    """
    try:
        logger.info("=== AIME Dataset Exploration ===")

        # Load AIME dataset using the data API with DataItem wrapper
        aime_data = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data("aime", streaming=False)
        ]
        logger.info(f"Successfully loaded {len(aime_data)} AIME problems")

        if not aime_data:
            logger.error("Dataset loaded but contains no problems")
            return

        # Sample problems
        sample_size = min(count, len(aime_data))
        logger.info(f"\nShowing {sample_size} sample AIME problems:")

        for i, problem in enumerate(aime_data[:sample_size]):
            # Clean access using DataItem properties
            problem_id = problem.problem_id if hasattr(problem, 'problem_id') else f"Problem {i+1}"
            logger.info(f"\n--- Problem {i+1}: {problem_id} ---")

            # Print problem query
            logger.info(f"Query: {problem.question}\n")

            # Print answer
            logger.info(f"Answer: {problem.answer}")

            # Print metadata
            logger.info("Metadata:")
            logger.info(f"  Year: {problem.year if hasattr(problem, 'year') else 'Unknown'}")
            logger.info(f"  Contest: {problem.contest if hasattr(problem, 'contest') else 'Unknown'}")

            if i < sample_size - 1:
                logger.info("\n" + "-" * 40)

        # Show filtering example
        logger.info("\n=== Filtering Example ===")

        # Filter to AIME I problems using builder
        aime_i = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("aime")
            .filter(lambda item: hasattr(item, 'contest') and item.contest == 'I')
            .build()
        ]
        logger.info(f"AIME I problems: {len(aime_i)}")

        # Filter to AIME II problems
        aime_ii = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("aime")
            .filter(lambda item: hasattr(item, 'contest') and item.contest == 'II')
            .build()
        ]
        logger.info(f"AIME II problems: {len(aime_ii)}")

    except Exception as e:
        logger.error(f"Error exploring AIME dataset: {e}")


def explore_gpqa(count: int = 3) -> None:
    """Explore GPQA dataset structure.

    Args:
        count: Number of examples to show
    """
    try:
        logger.info("=== GPQA Dataset Exploration ===")

        # Load GPQA dataset with DataItem wrapper
        gpqa_data = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data("gpqa", streaming=False)
        ]
        logger.info(f"Successfully loaded {len(gpqa_data)} GPQA questions")

        if not gpqa_data:
            logger.error("Dataset loaded but contains no questions")
            return

        # Sample questions
        sample_size = min(count, len(gpqa_data))
        logger.info(f"\nShowing {sample_size} sample GPQA questions:")

        for i, question in enumerate(gpqa_data[:sample_size]):
            question_id = question.id if hasattr(question, 'id') else f"Question {i+1}"
            logger.info(f"\n--- Question {i+1}: {question_id} ---")

            # Print question using clean DataItem access
            logger.info(f"Query: {question.question[:500]}...\n")

            # Print choices if available
            if question.options:
                logger.info("Choices:")
                for key, choice in question.options.items():
                    logger.info(
                        f"  {key}: {choice[:100]}..."
                        if len(choice) > 100
                        else f"  {key}: {choice}"
                    )

            # Print answer
            logger.info(f"\nCorrect Answer: {question.answer}")

            # Print metadata
            logger.info("Metadata:")
            logger.info(f"  Subject: {question.subject if hasattr(question, 'subject') else 'Unknown'}")
            logger.info(f"  Difficulty: {question.difficulty if hasattr(question, 'difficulty') else 'Unknown'}")

            if i < sample_size - 1:
                logger.info("\n" + "-" * 40)

        # Show filtering example
        logger.info("\n=== Filtering Example ===")

        # Filter by subject
        physics_questions = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("gpqa")
            .filter(lambda item: hasattr(item, 'subject') and "physics" in item.subject.lower())
            .build()
        ]
        logger.info(f"Physics questions: {len(physics_questions)}")

        chemistry_questions = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("gpqa")
            .filter(lambda item: hasattr(item, 'subject') and "chemistry" in item.subject.lower())
            .build()
        ]
        logger.info(f"Chemistry questions: {len(chemistry_questions)}")

    except Exception as e:
        logger.error(f"Error exploring GPQA dataset: {e}")
        if "gated dataset" in str(e) or "authentication" in str(e):
            logger.error("\nAuthentication required for GPQA dataset:")
            logger.error("1. Run: huggingface-cli login")
            logger.error(
                "2. Request access at: https://huggingface.co/datasets/Idavidrein/gpqa"
            )


def explore_codeforces(count: int = 3) -> None:
    """Explore Codeforces dataset structure.

    Args:
        count: Number of examples to show
    """
    try:
        logger.info("=== Codeforces Dataset Exploration ===")

        # Load Codeforces dataset with filtering and DataItem wrapper
        cf_data = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("codeforces")
            .config(difficulty_range=(800, 1200))  # Beginner-friendly
            .build()
        ]
        logger.info(f"Successfully loaded {len(cf_data)} Codeforces problems")

        if not cf_data:
            logger.error("Dataset loaded but contains no problems")
            return

        # Sample problems
        sample_size = min(count, len(cf_data))
        logger.info(f"\nShowing {sample_size} sample Codeforces problems:")

        for i, problem in enumerate(cf_data[:sample_size]):
            problem_id = problem.id if hasattr(problem, 'id') else f"Problem {i+1}"
            name = problem.name if hasattr(problem, 'name') else "Unnamed problem"
            logger.info(f"\n--- Problem {i+1}: {problem_id} - {name} ---")

            # Print problem using clean DataItem access
            query = problem.question
            query_preview = (
                query[:500] + "..."
                if len(query) > 500
                else query
            )
            logger.info(f"Query: {query_preview}\n")

            # Print metadata
            logger.info("Metadata:")
            logger.info(f"  Difficulty: {problem.difficulty if hasattr(problem, 'difficulty') else 'Unknown'}")
            logger.info(f"  Tags: {', '.join(problem.tags) if hasattr(problem, 'tags') and problem.tags else 'None'}")

            # Print test cases if available
            if hasattr(problem, 'test_cases') and problem.test_cases:
                test_cases = problem.test_cases
                logger.info(f"\nTest Cases ({len(test_cases)}):")
                for j, test in enumerate(test_cases[:2]):  # Show only first 2
                    logger.info(f"  Test {j+1}:")
                    if isinstance(test, dict):
                        logger.info(f"    Input: {test.get('input', 'N/A')}")
                        logger.info(f"    Expected Output: {test.get('output', 'N/A')}")

                if len(test_cases) > 2:
                    logger.info(f"  ... and {len(test_cases) - 2} more test cases")

            if i < sample_size - 1:
                logger.info("\n" + "-" * 40)

        # Show filtering example
        logger.info("\n=== Filtering Examples ===")

        # Filter by difficulty using config
        easy_problems = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("codeforces")
            .config(difficulty_range=(800, 1200))
            .build()
        ]
        logger.info(f"Easy problems (800-1200): {len(easy_problems)}")

        # Medium difficulty problems
        medium_problems = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("codeforces")
            .config(difficulty_range=(1300, 1700))
            .build()
        ]
        logger.info(f"Medium problems (1300-1700): {len(medium_problems)}")

        # Filter by tag
        dp_problems = [
            DataItem(item) if not isinstance(item, DataItem) else item
            for item in data.builder()
            .from_registry("codeforces")
            .config(tags=["dp"])
            .build()
        ]
        logger.info(f"Dynamic Programming problems: {len(dp_problems)}")

    except Exception as e:
        logger.error(f"Error exploring Codeforces dataset: {e}")


def main() -> None:
    """Explore specialized datasets in Ember."""
    parser = argparse.ArgumentParser(description="Explore Ember's specialized datasets")
    parser.add_argument(
        "--dataset",
        choices=["aime", "gpqa", "codeforces", "all"],
        default="all",
        help="Dataset to explore",
    )
    parser.add_argument(
        "--count", type=int, default=3, help="Number of examples to show"
    )
    args = parser.parse_args()

    # Explore requested datasets
    if args.dataset == "all" or args.dataset == "aime":
        explore_aime(args.count)
        if args.dataset != "all":
            return

    if args.dataset == "all" or args.dataset == "gpqa":
        if args.dataset == "all":
            logger.info("\n\n" + "=" * 50 + "\n")
        explore_gpqa(args.count)
        if args.dataset != "all":
            return

    if args.dataset == "all" or args.dataset == "codeforces":
        if args.dataset == "all":
            logger.info("\n\n" + "=" * 50 + "\n")
        explore_codeforces(args.count)


if __name__ == "__main__":
    main()