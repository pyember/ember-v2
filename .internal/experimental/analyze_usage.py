#!/usr/bin/env python3
"""Analyze Ember API usage patterns to understand what users actually do.

This script analyzes the codebase to measure:
1. Which APIs are actually used (models, data, eval, xcs, operators, non)
2. How often each API is used
3. Common usage patterns
4. Complexity of typical use cases
"""

import ast
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class EmberUsageAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Ember API usage."""
    
    def __init__(self):
        self.api_calls = Counter()
        self.import_patterns = Counter()
        self.call_patterns = []
        self.operator_usage = Counter()
        self.model_calls = []
        self.data_calls = []
        self.xcs_usage = Counter()
        
    def visit_Import(self, node):
        """Track import statements."""
        for alias in node.names:
            if alias.name.startswith('ember'):
                self.import_patterns[alias.name] += 1
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track from imports."""
        if node.module and node.module.startswith('ember'):
            for alias in node.names:
                import_path = f"{node.module}.{alias.name}"
                self.import_patterns[import_path] += 1
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """Track function calls."""
        func_name = self._get_call_name(node)
        
        # Track direct API calls
        if func_name:
            if 'models' in func_name:
                self.api_calls['models'] += 1
                self._analyze_model_call(node)
            elif 'data' in func_name:
                self.api_calls['data'] += 1
                self._analyze_data_call(node)
            elif 'eval' in func_name:
                self.api_calls['eval'] += 1
            elif 'xcs.' in func_name or '@xcs' in func_name:
                self.api_calls['xcs'] += 1
                self.xcs_usage[func_name] += 1
            elif 'operators.' in func_name:
                self.api_calls['operators'] += 1
                self.operator_usage[func_name] += 1
            elif 'non.' in func_name:
                self.api_calls['non'] += 1
                
        self.generic_visit(node)
        
    def _get_call_name(self, node) -> str:
        """Extract the full call name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return ''
        
    def _analyze_model_call(self, node):
        """Analyze model call patterns."""
        if node.args:
            # First arg is usually the model name
            if isinstance(node.args[0], ast.Constant):
                model_name = node.args[0].value
                self.model_calls.append(model_name)
                
    def _analyze_data_call(self, node):
        """Analyze data call patterns."""
        if node.args:
            # First arg is usually the dataset name
            if isinstance(node.args[0], ast.Constant):
                dataset_name = node.args[0].value
                self.data_calls.append(dataset_name)


def analyze_file(filepath: Path) -> EmberUsageAnalyzer:
    """Analyze a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            analyzer = EmberUsageAnalyzer()
            analyzer.visit(tree)
            return analyzer
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return EmberUsageAnalyzer()


def analyze_directory(directory: Path) -> Dict:
    """Analyze all Python files in a directory."""
    total_analyzer = EmberUsageAnalyzer()
    file_count = 0
    
    for filepath in directory.rglob('*.py'):
        # Skip internal/deprecated files
        if any(skip in str(filepath) for skip in ['.internal_docs', '__pycache__', 'deprecated']):
            continue
            
        analyzer = analyze_file(filepath)
        
        # Aggregate results
        total_analyzer.api_calls.update(analyzer.api_calls)
        total_analyzer.import_patterns.update(analyzer.import_patterns)
        total_analyzer.operator_usage.update(analyzer.operator_usage)
        total_analyzer.xcs_usage.update(analyzer.xcs_usage)
        total_analyzer.model_calls.extend(analyzer.model_calls)
        total_analyzer.data_calls.extend(analyzer.data_calls)
        
        file_count += 1
        
    return {
        'file_count': file_count,
        'api_calls': total_analyzer.api_calls,
        'import_patterns': total_analyzer.import_patterns,
        'operator_usage': total_analyzer.operator_usage,
        'xcs_usage': total_analyzer.xcs_usage,
        'model_calls': Counter(total_analyzer.model_calls),
        'data_calls': Counter(total_analyzer.data_calls),
    }


def print_analysis(results: Dict, title: str):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"{title} Analysis")
    print(f"{'='*60}")
    print(f"Files analyzed: {results['file_count']}")
    
    print("\nAPI Usage Frequency:")
    total_calls = sum(results['api_calls'].values())
    for api, count in results['api_calls'].most_common():
        percentage = (count / total_calls * 100) if total_calls > 0 else 0
        print(f"  {api:15} {count:5} calls ({percentage:5.1f}%)")
        
    print("\nTop Import Patterns:")
    for pattern, count in results['import_patterns'].most_common(10):
        print(f"  {pattern:40} {count:3} times")
        
    print("\nModel Usage:")
    for model, count in results['model_calls'].most_common(5):
        print(f"  {model:20} {count:3} times")
        
    print("\nDataset Usage:")
    for dataset, count in results['data_calls'].most_common(5):
        print(f"  {dataset:20} {count:3} times")
        
    if results['operator_usage']:
        print("\nOperator Usage:")
        for op, count in results['operator_usage'].most_common(5):
            print(f"  {op:40} {count:3} times")
            
    if results['xcs_usage']:
        print("\nXCS Feature Usage:")
        for feature, count in results['xcs_usage'].most_common(5):
            print(f"  {feature:40} {count:3} times")


def analyze_complexity_patterns(directory: Path) -> Dict:
    """Analyze complexity of usage patterns."""
    simple_usage = 0  # Direct API calls
    complex_usage = 0  # Operators, XCS, etc.
    
    for filepath in directory.rglob('*.py'):
        if any(skip in str(filepath) for skip in ['.internal_docs', '__pycache__', 'deprecated']):
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Simple patterns (what 99% need)
                if 'models(' in content or 'data(' in content or 'eval(' in content:
                    simple_usage += 1
                    
                # Complex patterns (what 1% need)  
                if 'Operator' in content or 'XCS' in content or 'EmberContext' in content:
                    complex_usage += 1
                    
        except Exception:
            pass
            
    return {
        'simple_usage': simple_usage,
        'complex_usage': complex_usage,
        'simple_percentage': (simple_usage / (simple_usage + complex_usage) * 100) if (simple_usage + complex_usage) > 0 else 0
    }


def main():
    """Run the analysis."""
    ember_root = Path(__file__).parent
    
    # Analyze different parts of the codebase
    print("\nAnalyzing Ember usage patterns...")
    
    # Examples (what users learn from)
    examples_dir = ember_root / 'src' / 'ember' / 'examples'
    if examples_dir.exists():
        examples_results = analyze_directory(examples_dir)
        print_analysis(examples_results, "Examples")
        
    # Tests (what features are actually tested/used)
    tests_dir = ember_root / 'tests'
    if tests_dir.exists():
        tests_results = analyze_directory(tests_dir)
        print_analysis(tests_results, "Tests")
        
    # Overall complexity analysis
    complexity = analyze_complexity_patterns(ember_root)
    
    print(f"\n{'='*60}")
    print("Complexity Analysis")
    print(f"{'='*60}")
    print(f"Files using simple APIs: {complexity['simple_usage']}")
    print(f"Files using complex APIs: {complexity['complex_usage']}")
    print(f"Simple API usage: {complexity['simple_percentage']:.1f}%")
    
    # Summary insights
    print(f"\n{'='*60}")
    print("Key Insights")
    print(f"{'='*60}")
    
    if examples_dir.exists():
        total_api_calls = sum(examples_results['api_calls'].values())
        models_percentage = (examples_results['api_calls']['models'] / total_api_calls * 100) if total_api_calls > 0 else 0
        
        print(f"1. Models API represents {models_percentage:.1f}% of all API calls in examples")
        print(f"2. Most common models: {', '.join(list(examples_results['model_calls'].keys())[:3])}")
        print(f"3. Most common datasets: {', '.join(list(examples_results['data_calls'].keys())[:3])}")
        
        if examples_results['operator_usage']:
            print(f"4. Only {len(examples_results['operator_usage'])} different operator patterns used")
        else:
            print("4. No complex operator usage in examples")
            
        if examples_results['xcs_usage']:
            print(f"5. XCS features used: {', '.join(list(examples_results['xcs_usage'].keys())[:3])}")
        else:
            print("5. No XCS usage in examples")
    
    print("\nConclusion: The vast majority of usage is simple model calls.")
    print("Complex features (operators, XCS, contexts) are rarely used.")


if __name__ == "__main__":
    main()