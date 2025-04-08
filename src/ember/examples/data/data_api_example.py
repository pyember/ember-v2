"""Data API Example.

A demonstration of Ember's data system using the MMLU dataset.
Shows how to load, transform, and analyze dataset entries
with proper configuration and error handling.
"""

import logging
from typing import Dict, Any, List, Optional

from ember.core.utils.data.context.data_context import DataContext
from ember.core.utils.data import load_dataset_entries
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.datasets_registry.mmlu import MMLUConfig


def load_mmlu_questions(subject: str = "high_school_mathematics", 
                       split: str = "test",
                       num_samples: int = 5) -> List[DatasetEntry]:
    """Load questions from MMLU dataset with proper configuration.
    
    Args:
        subject: Subject area to load
        split: Dataset split to use (test, validation, etc.)
        num_samples: Maximum number of questions to load
        
    Returns:
        List of dataset entries
    
    Raises:
        RuntimeError: If dataset loading fails
    """
    # Create data context
    context = DataContext(auto_discover=True)
    
    try:
        # MMLU requires specific config_name and split parameters
        config = MMLUConfig(config_name=subject, split=split)
        
        entries = load_dataset_entries(
            dataset_name="mmlu",
            config=config,
            num_samples=num_samples,
            context=context
        )
        return entries
    except Exception as e:
        logging.error(f"Failed to load MMLU dataset: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}") from e


def print_question_details(entries: List[DatasetEntry]) -> None:
    """Display formatted questions with answers and options.
    
    Args:
        entries: List of dataset entries to display
    """
    for i, entry in enumerate(entries, 1):
        print(f"\nQuestion {i}: {entry.query}")
        
        if hasattr(entry, "choices") and entry.choices:
            print("\nOptions:")
            for key, text in sorted(entry.choices.items()):
                print(f"  {key}) {text}")
            
        if hasattr(entry, "metadata") and entry.metadata.get("correct_answer"):
            print(f"\nAnswer: {entry.metadata['correct_answer']}")
        
        print("-" * 80)


def transform_to_prompt_format(entries: List[DatasetEntry]) -> List[Dict[str, Any]]:
    """Transform dataset entries to LLM-ready prompt format.
    
    Args:
        entries: Dataset entries to transform
        
    Returns:
        List of formatted prompt dictionaries
    """
    transformed = []
    
    for entry in entries:
        # Format choices as a string
        options_text = ""
        if hasattr(entry, "choices") and entry.choices:
            options = []
            for key, text in sorted(entry.choices.items()):
                options.append(f"{key}. {text}")
            options_text = "\n".join(options)
        
        # Build formatted prompt with metadata
        prompt = {
            "formatted_question": f"Question: {entry.query}\n\n{options_text}",
            "answer": entry.metadata.get("correct_answer", "") if hasattr(entry, "metadata") else "",
            "subject": entry.metadata.get("subject", "") if hasattr(entry, "metadata") else "",
        }
        
        transformed.append(prompt)
    
    return transformed


def analyze_dataset(entries: List[DatasetEntry]) -> Dict[str, Any]:
    """Analyze dataset composition and structure.
    
    Args:
        entries: Dataset entries to analyze
        
    Returns:
        Dictionary of dataset statistics
    """
    stats = {
        "total_entries": len(entries),
        "avg_choices": 0,
        "choices_distribution": {},
        "query_length": {"min": float('inf'), "max": 0, "avg": 0},
    }
    
    total_choices = 0
    total_query_length = 0
    
    for entry in entries:
        # Count choices
        if hasattr(entry, "choices"):
            num_choices = len(entry.choices)
            total_choices += num_choices
            
            if num_choices not in stats["choices_distribution"]:
                stats["choices_distribution"][num_choices] = 0
            stats["choices_distribution"][num_choices] += 1
        
        # Measure query length
        query_length = len(entry.query)
        total_query_length += query_length
        stats["query_length"]["min"] = min(stats["query_length"]["min"], query_length)
        stats["query_length"]["max"] = max(stats["query_length"]["max"], query_length)
    
    # Calculate averages
    if stats["total_entries"] > 0:
        stats["avg_choices"] = total_choices / stats["total_entries"]
        stats["query_length"]["avg"] = total_query_length / stats["total_entries"]
    
    return stats


def main() -> None:
    """Run the MMLU data example."""
    # Configure structured logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("EMBER DATA API EXAMPLE")
    print("=====================")
    
    try:
        # Load MMLU datasets from different subjects
        math_entries = load_mmlu_questions(
            subject="high_school_mathematics", 
            num_samples=3
        )
        print(f"Loaded {len(math_entries)} MMLU math questions")
        
        biology_entries = load_mmlu_questions(
            subject="high_school_biology", 
            num_samples=2
        )
        print(f"Loaded {len(biology_entries)} MMLU biology questions")
        
        # Combine entries
        entries = math_entries + biology_entries
        print(f"Total: {len(entries)} questions from multiple subjects\n")
        
        # Display formatted questions
        print_question_details(entries)
        
        # Transform to prompt format
        formatted = transform_to_prompt_format(entries)
        
        # Display an example of transformed format
        print("\nTRANSFORMED PROMPT FORMAT (EXAMPLE)")
        print("=================================")
        if formatted:
            print(formatted[0]["formatted_question"])
            print(f"\nExpected answer: {formatted[0]['answer']}")
        
        # Analyze dataset composition
        stats = analyze_dataset(entries)
        print("\nDATASET ANALYSIS")
        print("===============")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Average choices per question: {stats['avg_choices']:.1f}")
        print(f"Question length: min={stats['query_length']['min']}, " +
              f"max={stats['query_length']['max']}, " +
              f"avg={stats['query_length']['avg']:.1f} characters")
        
    except Exception as e:
        logging.error(f"Example failed: {e}")
        print(f"\nError: {e}")
        print("\nPlease ensure you have the necessary dataset access and environment setup.")


if __name__ == "__main__":
    main()