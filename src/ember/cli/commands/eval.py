"""Eval command implementation."""

import argparse
import sys
from typing import Optional, List, Dict, Any

from ember.core.utils.output import print_header, print_table, print_info, print_error
from ember.core.utils.progress import ProgressReporter
from ember.core.utils.verbosity import get_verbosity
from ember.api import data, models
from ember.api import eval as eval_api


def register(subparsers) -> argparse.ArgumentParser:
    """Add eval command to subparsers."""
    parser = subparsers.add_parser(
        "eval",
        help="Run evaluations on models",
        description="Evaluate models on datasets with various metrics"
    )
    
    parser.add_argument("model", help="Model to evaluate")
    parser.add_argument("dataset", help="Dataset to use for evaluation")
    
    # Optional arguments
    parser.add_argument(
        "--evaluator",
        default="accuracy",
        help="Evaluator to use (default: accuracy)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples to evaluate"
    )
    parser.add_argument(
        "--output",
        help="Save results to file"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format"
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show examples where the model was incorrect"
    )
    
    parser.set_defaults(func=execute)
    return parser


def execute(args: argparse.Namespace) -> int:
    """Execute eval command."""
    progress = ProgressReporter()
    
    try:
        # Validate model
        available_models = models.available()
        if args.model not in available_models:
            print_error(f"Model '{args.model}' not found")
            return 1
        
        # Validate dataset
        available_datasets = data.list()
        if args.dataset not in available_datasets:
            print_error(f"Dataset '{args.dataset}' not found")
            print("\nAvailable datasets:")
            for d in sorted(available_datasets):
                print(f"  - {d}")
            return 1
        
        # Validate evaluator
        available_evaluators = eval_api.list_available_evaluators()
        if args.evaluator not in available_evaluators:
            print_error(f"Evaluator '{args.evaluator}' not found")
            print("\nAvailable evaluators:")
            for e in sorted(available_evaluators):
                print(f"  - {e}")
            return 1
        
        print_header(f"Evaluating {args.model} on {args.dataset}")
        print(f"Evaluator: {args.evaluator}")
        
        # Load dataset
        if get_verbosity() >= 1:
            progress.loading_start("dataset")
        
        dataset = data(args.dataset, streaming=True, limit=args.limit)
        if args.limit:
            print(f"Limited to {args.limit} examples")
        
        if get_verbosity() >= 1:
            progress.loading_complete("dataset")
        
        # Get model and evaluator
        model = models(args.model)
        evaluator = eval_api.Evaluator.from_registry(args.evaluator)
        
        # Run evaluation
        if get_verbosity() >= 1:
            progress.execution_start("evaluation")
        
        results = []
        errors = []
        correct = 0
        total = 0
        
        for example in dataset:
            total += 1
            
            # Generate model response
            response = model(example["prompt"])
            
            # Evaluate
            score = evaluator(response, example.get("answer", example.get("expected")))
            
            if score > 0.5:  # Assuming binary accuracy
                correct += 1
            elif args.show_errors:
                errors.append({
                    "prompt": example["prompt"][:50] + "..." if len(example["prompt"]) > 50 else example["prompt"],
                    "expected": example.get("answer", example.get("expected")),
                    "got": response[:50] + "..." if len(response) > 50 else response
                })
            
            # Progress update
            if total % 10 == 0 and get_verbosity() >= 1:
                print(f"  Evaluated {total} examples... (accuracy: {correct/total:.2%})")
        
        if get_verbosity() >= 1:
            progress.execution_complete()
        
        # Calculate final metrics
        accuracy = correct / total if total > 0 else 0
        
        # Display results
        print_header("Results")
        
        if args.format == "json":
            import json
            output = {
                "model": args.model,
                "dataset": args.dataset,
                "evaluator": args.evaluator,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            }
            if args.show_errors:
                output["errors"] = errors
            print(json.dumps(output, indent=2))
        
        elif args.format == "csv":
            print("model,dataset,evaluator,total,correct,accuracy")
            print(f"{args.model},{args.dataset},{args.evaluator},{total},{correct},{accuracy:.4f}")
        
        else:  # table format
            metrics = [
                {"Metric": "Total Examples", "Value": str(total)},
                {"Metric": "Correct", "Value": str(correct)},
                {"Metric": "Accuracy", "Value": f"{accuracy:.2%}"}]
            print_table(metrics, columns=["Metric", "Value"])
            
            if args.show_errors and errors:
                print_header("Errors (first 10)")
                print_table(errors[:10])
        
        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                if args.format == "json":
                    import json
                    json.dump({
                        "model": args.model,
                        "dataset": args.dataset,
                        "evaluator": args.evaluator,
                        "total": total,
                        "correct": correct,
                        "accuracy": accuracy,
                        "errors": errors if args.show_errors else []
                    }, f, indent=2)
                else:
                    f.write(f"Model: {args.model}\n")
                    f.write(f"Dataset: {args.dataset}\n")
                    f.write(f"Evaluator: {args.evaluator}\n")
                    f.write(f"Total: {total}\n")
                    f.write(f"Correct: {correct}\n")
                    f.write(f"Accuracy: {accuracy:.2%}\n")
            
            print_info(f"Results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print_error(f"Error during evaluation: {e}")
        return 1