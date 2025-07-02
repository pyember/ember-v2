"""Accuracy Evaluation - Measure AI system performance.

Build evaluation frameworks with metrics, statistical analysis,
and performance tracking for compound AI systems.

Example:
    >>> evaluator = SystemEvaluator(metrics=["accuracy", "f1"])
    >>> results = evaluator(predictions=preds, ground_truth=labels)
    >>> print(f"Accuracy: {results['accuracy']:.2%}")
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import statistics

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output, timer
from ember.api import operators


def main():
    """Example demonstrating the simplified XCS architecture."""
    """Build evaluation frameworks for AI systems."""
    print_section_header("Accuracy Evaluation Framework")
    
    # Part 1: Create Test Dataset
    print("📊 Part 1: Creating Evaluation Datasets\n")
    
    class EvaluationDataset(operators.Operator):
        """Creates test datasets with ground truth."""
        
        def forward(self, inputs):
            task_type = inputs.get("task_type", "classification")
            
            if task_type == "classification":
                # Simple sentiment classification dataset
                test_cases = [
                    {"text": "This product is amazing! Best purchase ever!", "label": "positive", "difficulty": "easy"},
                    {"text": "Terrible quality, waste of money.", "label": "negative", "difficulty": "easy"},
                    {"text": "It's okay, nothing special but works.", "label": "neutral", "difficulty": "medium"},
                    {"text": "Mixed feelings - great features but poor build.", "label": "neutral", "difficulty": "hard"},
                    {"text": "Exceeded expectations in every way!", "label": "positive", "difficulty": "easy"},
                    {"text": "Disappointing experience overall.", "label": "negative", "difficulty": "medium"},
                    {"text": "Decent value for the price point.", "label": "neutral", "difficulty": "medium"},
                    {"text": "Revolutionary product that changes everything!", "label": "positive", "difficulty": "easy"},
                    {"text": "Complete disaster, nothing works.", "label": "negative", "difficulty": "easy"},
                    {"text": "Has pros and cons, depends on use case.", "label": "neutral", "difficulty": "hard"}]
            elif task_type == "math":
                # Simple math problems
                test_cases = [
                    {"question": "What is 15 + 27?", "answer": 42, "difficulty": "easy"},
                    {"question": "What is 8 × 7?", "answer": 56, "difficulty": "easy"},
                    {"question": "What is 144 ÷ 12?", "answer": 12, "difficulty": "medium"},
                    {"question": "What is 23 × 17?", "answer": 391, "difficulty": "medium"},
                    {"question": "What is sqrt(169)?", "answer": 13, "difficulty": "medium"},
                    {"question": "What is 2^8?", "answer": 256, "difficulty": "medium"},
                    {"question": "What is 15% of 80?", "answer": 12, "difficulty": "hard"},
                    {"question": "What is 7! (factorial)?", "answer": 5040, "difficulty": "hard"}]
            else:
                test_cases = []
            
            return {
                "test_cases": test_cases,
                "task_type": task_type,
                "total_cases": len(test_cases),
                "difficulty_distribution": self.get_difficulty_distribution(test_cases)
            }
        
        def get_difficulty_distribution(self, test_cases):
            distribution = {"easy": 0, "medium": 0, "hard": 0}
            for case in test_cases:
                distribution[case.get("difficulty", "medium")] += 1
            return distribution
    
    # Create datasets
    dataset_creator = EvaluationDataset()
    
    classification_data = dataset_creator({"task_type": "classification"})
    math_data = dataset_creator({"task_type": "math"})
    
    print(f"Created classification dataset: {classification_data['total_cases']} cases")
    print(f"Difficulty: {classification_data['difficulty_distribution']}")
    print(f"\nCreated math dataset: {math_data['total_cases']} cases")
    print(f"Difficulty: {math_data['difficulty_distribution']}")
    
    # Part 2: System Under Test
    print("\n" + "="*50)
    print("🤖 Part 2: Systems to Evaluate")
    print("="*50 + "\n")
    
    class SimpleSentimentClassifier(operators.Operator):
        """Simple rule-based sentiment classifier."""
        
        def forward(self, inputs):
            text = inputs.get("text", "").lower()
            
            positive_words = ["amazing", "great", "excellent", "love", "best", "exceeded", "revolutionary"]
            negative_words = ["terrible", "waste", "poor", "disappointing", "disaster", "nothing works"]
            neutral_phrases = ["okay", "decent", "pros and cons", "mixed", "depends"]
            
            pos_score = sum(1 for word in positive_words if word in text)
            neg_score = sum(1 for word in negative_words if word in text)
            neutral_score = sum(1 for phrase in neutral_phrases if phrase in text)
            
            if neutral_score > 0 or (pos_score > 0 and neg_score > 0):
                prediction = "neutral"
            elif pos_score > neg_score:
                prediction = "positive"
            elif neg_score > pos_score:
                prediction = "negative"
            else:
                prediction = "neutral"  # default
            
            confidence = max(pos_score, neg_score, neutral_score, 1) / (pos_score + neg_score + neutral_score + 1)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "scores": {"positive": pos_score, "negative": neg_score, "neutral": neutral_score}
            }
    
    class SimpleMathSolver(operators.Operator):
        """Simple math problem solver."""
        
        
        def forward(self, inputs):
            question = inputs.get("question", "")
            
            # Parse and solve (simplified for demo)
            try:
                if "+" in question:
                    numbers = [int(x) for x in question.split() if x.isdigit()]
                    if len(numbers) == 2:
                        answer = numbers[0] + numbers[1]
                    else:
                        answer = None
                elif "×" in question or "*" in question:
                    numbers = [int(x) for x in question.split() if x.isdigit()]
                    if len(numbers) == 2:
                        answer = numbers[0] * numbers[1]
                    else:
                        answer = None
                elif "÷" in question or "/" in question:
                    numbers = [int(x) for x in question.split() if x.isdigit()]
                    if len(numbers) == 2 and numbers[1] != 0:
                        answer = numbers[0] / numbers[1]
                    else:
                        answer = None
                else:
                    # Random guess for demo
                    answer = random.randint(1, 100)
                
                return {
                    "answer": answer,
                    "confidence": 0.8 if answer else 0.2,
                    "method": "parsed" if answer else "guessed"
                }
            except:
                return {
                    "answer": random.randint(1, 100),
                    "confidence": 0.1,
                    "method": "failed"
                }
    
    # Test the systems
    classifier = SimpleSentimentClassifier()
    math_solver = SimpleMathSolver()
    
    print("Testing sentiment classifier:")
    test_text = "This product is amazing but has some issues."
    result = classifier({"text": test_text})
    print(f"Text: '{test_text}'")
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
    
    print("\nTesting math solver:")
    test_question = "What is 15 + 27?"
    result = math_solver({"question": test_question})
    print(f"Question: '{test_question}'")
    print(f"Answer: {result['answer']} (method: {result['method']})")
    
    # Part 3: Evaluation Metrics
    print("\n" + "="*50)
    print("📏 Part 3: Evaluation Metrics")
    print("="*50 + "\n")
    
    class AccuracyEvaluator(operators.Operator):
        """Calculates accuracy metrics for classification."""
        
        
        def forward(self, inputs):
            predictions = inputs.get("predictions", [])
            ground_truth = inputs.get("ground_truth", [])
            
            if len(predictions) != len(ground_truth):
                return {"error": "Predictions and ground truth must have same length"}
            
            # Calculate metrics
            correct = sum(1 for pred, true in zip(predictions, ground_truth) if pred == true)
            total = len(predictions)
            accuracy = correct / total if total > 0 else 0
            
            # Per-class metrics
            classes = list(set(ground_truth))
            class_metrics = {}
            
            for cls in classes:
                true_positives = sum(1 for p, t in zip(predictions, ground_truth) if p == cls and t == cls)
                false_positives = sum(1 for p, t in zip(predictions, ground_truth) if p == cls and t != cls)
                false_negatives = sum(1 for p, t in zip(predictions, ground_truth) if p != cls and t == cls)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[cls] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": sum(1 for t in ground_truth if t == cls)
                }
            
            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "class_metrics": class_metrics,
                "macro_f1": statistics.mean([m["f1"] for m in class_metrics.values()])
            }
    
    # Part 4: Complete Evaluation Pipeline
    print("\n" + "="*50)
    print("🔬 Part 4: Complete Evaluation Pipeline")
    print("="*50 + "\n")
    
    class EvaluationPipeline(operators.Operator):
        """Complete evaluation pipeline for any system."""
        
        
        def __init__(self, *, system: operators.Operator, evaluator: operators.Operator):
            self.system = system
            self.evaluator = evaluator
        
        def forward(self, inputs):
            dataset = inputs.get("dataset", [])
            task_type = inputs.get("task_type", "classification")
            
            # Run predictions
            predictions = []
            ground_truth = []
            results_by_difficulty = {"easy": [], "medium": [], "hard": []}
            
            with timer("Running predictions"):
                for case in dataset:
                    # Get prediction
                    if task_type == "classification":
                        result = self.system({"text": case["text"]})
                        prediction = result["prediction"]
                        truth = case["label"]
                    elif task_type == "math":
                        result = self.system(question=case["question"])
                        prediction = result["answer"]
                        truth = case["answer"]
                    else:
                        continue
                    
                    predictions.append(prediction)
                    ground_truth.append(truth)
                    
                    # Track by difficulty
                    difficulty = case.get("difficulty", "medium")
                    is_correct = prediction == truth
                    results_by_difficulty[difficulty].append(is_correct)
            
            # Calculate metrics
            if task_type == "classification":
                metrics = self.evaluator({"predictions": predictions, "ground_truth": ground_truth})
            else:
                # For math, simple accuracy
                correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
                metrics = {
                    "accuracy": correct / len(predictions) if predictions else 0,
                    "correct": correct,
                    "total": len(predictions)
                }
            
            # Calculate per-difficulty accuracy
            difficulty_accuracy = {}
            for difficulty, results in results_by_difficulty.items():
                if results:
                    difficulty_accuracy[difficulty] = sum(results) / len(results)
                else:
                    difficulty_accuracy[difficulty] = 0.0
            
            return {
                "overall_metrics": metrics,
                "difficulty_accuracy": difficulty_accuracy,
                "task_type": task_type,
                "num_samples": len(dataset)
            }
    
    # Run evaluations
    eval_pipeline_classification = EvaluationPipeline(
        system=classifier,
        evaluator=AccuracyEvaluator()
    )
    
    eval_pipeline_math = EvaluationPipeline(
        system=math_solver,
        evaluator=AccuracyEvaluator()
    )
    
    print("Running Classification Evaluation:")
    classification_results = eval_pipeline_classification(
        dataset=classification_data["test_cases"],
        task_type="classification"
    )
    
    print(f"\nOverall Accuracy: {classification_results['overall_metrics']['accuracy']:.2%}")
    print(f"Correct: {classification_results['overall_metrics']['correct']}/{classification_results['overall_metrics']['total']}")
    print("\nPer-difficulty accuracy:")
    for difficulty, acc in classification_results['difficulty_accuracy'].items():
        print(f"  {difficulty}: {acc:.2%}")
    
    if "class_metrics" in classification_results['overall_metrics']:
        print("\nPer-class metrics:")
        for cls, metrics in classification_results['overall_metrics']['class_metrics'].items():
            print(f"  {cls}: F1={metrics['f1']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}")
    
    print("\n" + "-"*30 + "\n")
    print("Running Math Evaluation:")
    math_results = eval_pipeline_math(
        dataset=math_data["test_cases"],
        task_type="math"
    )
    
    print(f"\nOverall Accuracy: {math_results['overall_metrics']['accuracy']:.2%}")
    print(f"Correct: {math_results['overall_metrics']['correct']}/{math_results['overall_metrics']['total']}")
    print("\nPer-difficulty accuracy:")
    for difficulty, acc in math_results['difficulty_accuracy'].items():
        print(f"  {difficulty}: {acc:.2%}")
    
    # Part 5: Comparison Framework
    print("\n" + "="*50)
    print("⚖️ Part 5: System Comparison")
    print("="*50 + "\n")
    
    class ComparisonEvaluator(operators.Operator):
        """Compares multiple systems on same dataset."""
        
        
        def __init__(self, *, systems: Dict[str, operators.Operator]):
            self.systems = systems
            self.pipeline = {}
            for name, system in systems.items():
                self.pipeline[name] = EvaluationPipeline(
                    system=system,
                    evaluator=AccuracyEvaluator()
                )
        
        def forward(self, inputs):
            dataset = inputs.get("dataset", [])
            task_type = inputs.get("task_type", "classification")
            
            results = {}
            rankings = []
            
            for name, pipeline in self.pipeline.items():
                result = pipeline({"dataset": dataset, "task_type": task_type})
                results[name] = result
                rankings.append((name, result['overall_metrics']['accuracy']))
            
            # Sort by accuracy
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            # Statistical significance (simplified)
            if len(rankings) >= 2:
                best_acc = rankings[0][1]
                second_acc = rankings[1][1]
                n_samples = len(dataset)
                
                # Simple significance test
                margin = 1.96 * ((best_acc * (1 - best_acc)) / n_samples) ** 0.5
                is_significant = (best_acc - second_acc) > margin
            else:
                is_significant = False
            
            return {
                "results": results,
                "rankings": rankings,
                "winner": rankings[0][0] if rankings else None,
                "is_significant": is_significant,
                "task_type": task_type
            }
    
    # Create alternative classifier
    class RandomClassifier(operators.Operator):
        """Baseline random classifier."""
        
        def forward(self, inputs):
            return {
                "prediction": random.choice(["positive", "negative", "neutral"]),
                "confidence": random.random()
            }
    
    # Compare systems
    comparator = ComparisonEvaluator(systems={
        "rule_based": classifier,
        "random_baseline": RandomClassifier()
    })
    
    comparison = comparator(
        dataset=classification_data["test_cases"],
        task_type="classification"
    )
    
    print("System Comparison Results:")
    print("\nRankings:")
    for i, (name, accuracy) in enumerate(comparison['rankings'], 1):
        print(f"{i}. {name}: {accuracy:.2%}")
    
    print(f"\nWinner: {comparison['winner']}")
    print(f"Statistically significant: {'Yes' if comparison['is_significant'] else 'No'}")
    
    print("\n" + "="*50)
    print("✅ Key Takeaways")
    print("="*50)
    print("\n1. Always evaluate on representative test sets")
    print("2. Use multiple metrics (accuracy, F1, precision, recall)")
    print("3. Consider difficulty stratification")
    print("4. Compare against baselines")
    print("5. Check statistical significance")
    print("6. Track performance over time")
    print("7. Automate evaluation pipelines")
    
    print("\nNext: Explore consistency_testing.py for reliability metrics!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())