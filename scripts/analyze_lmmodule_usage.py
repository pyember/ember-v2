#!/usr/bin/env python3
"""Analyze LMModule usage across the codebase.

This script helps identify all places where LMModule is used and categorizes
them by risk and complexity for migration planning.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json


class LMModuleAnalyzer(ast.NodeVisitor):
    """AST visitor to find LMModule usage patterns."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports: List[Dict] = []
        self.instantiations: List[Dict] = []
        self.method_calls: List[Dict] = []
        self.in_test = "test" in filepath
        
    def visit_ImportFrom(self, node):
        """Find imports of LMModule or LMModuleConfig."""
        if node.module and "lm" in node.module:
            for alias in node.names:
                if alias.name in ["LMModule", "LMModuleConfig"]:
                    self.imports.append({
                        "line": node.lineno,
                        "name": alias.name,
                        "alias": alias.asname,
                        "module": node.module
                    })
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Find instantiations and method calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in ["LMModule", "LMModuleConfig"]:
                self.instantiations.append({
                    "line": node.lineno,
                    "class": node.func.id,
                    "args": len(node.args),
                    "kwargs": [kw.arg for kw in node.keywords]
                })
        elif isinstance(node.func, ast.Attribute):
            # Check for method calls on potential LMModule instances
            if node.func.attr in ["__call__", "forward", "generate"]:
                self.method_calls.append({
                    "line": node.lineno,
                    "method": node.func.attr,
                    "context": ast.unparse(node.func.value) if hasattr(ast, 'unparse') else "unknown"
                })
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single Python file for LMModule usage."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))
        
        analyzer = LMModuleAnalyzer(str(filepath))
        analyzer.visit(tree)
        
        return {
            "file": str(filepath),
            "imports": analyzer.imports,
            "instantiations": analyzer.instantiations,
            "method_calls": analyzer.method_calls,
            "is_test": analyzer.in_test,
            "total_usage": len(analyzer.imports) + len(analyzer.instantiations)
        }
    except Exception as e:
        return {
            "file": str(filepath),
            "error": str(e),
            "total_usage": 0
        }


def calculate_risk_score(analysis: Dict) -> Tuple[int, str]:
    """Calculate migration risk score for a file."""
    score = 0
    factors = []
    
    # Base scoring
    score += len(analysis.get("instantiations", [])) * 2
    score += len(analysis.get("method_calls", []))
    
    # Risk factors
    if "operator" in analysis["file"].lower():
        score += 5
        factors.append("operator file")
    
    if "ensemble" in analysis["file"].lower():
        score += 10
        factors.append("ensemble operator")
    
    if analysis.get("total_usage", 0) > 5:
        score += 5
        factors.append("heavy usage")
    
    if not analysis.get("is_test", False) and score > 0:
        score += 3
        factors.append("production code")
    
    # Risk level
    if score == 0:
        risk = "NONE"
    elif score < 5:
        risk = "LOW"
    elif score < 15:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
    
    return score, risk, factors


def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files in the project."""
    python_files = []
    exclude_dirs = {".git", "__pycache__", ".tox", "venv", "env", ".venv"}
    
    for path in root_dir.rglob("*.py"):
        if not any(excluded in path.parts for excluded in exclude_dirs):
            python_files.append(path)
    
    return python_files


def generate_migration_order(analyses: List[Dict]) -> List[Dict]:
    """Generate recommended migration order based on dependencies and risk."""
    # Sort by risk score and dependencies
    scored = []
    for analysis in analyses:
        if analysis.get("total_usage", 0) > 0:
            score, risk, factors = calculate_risk_score(analysis)
            analysis["risk_score"] = score
            analysis["risk_level"] = risk
            analysis["risk_factors"] = factors
            scored.append(analysis)
    
    # Sort by risk (ascending) and whether it's a test file
    scored.sort(key=lambda x: (x.get("is_test", False), x["risk_score"]))
    
    return scored


def main():
    """Main analysis function."""
    root_dir = Path.cwd()
    print(f"Analyzing LMModule usage in: {root_dir}")
    print("=" * 80)
    
    # Find and analyze all Python files
    python_files = find_python_files(root_dir / "src")
    python_files.extend(find_python_files(root_dir / "tests"))
    
    analyses = []
    for filepath in python_files:
        analysis = analyze_file(filepath)
        if analysis.get("total_usage", 0) > 0 or analysis.get("error"):
            analyses.append(analysis)
    
    # Generate migration order
    migration_order = generate_migration_order(analyses)
    
    # Print summary
    print(f"\nFound {len(migration_order)} files using LMModule")
    print("\nRisk Distribution:")
    risk_counts = defaultdict(int)
    for item in migration_order:
        risk_counts[item.get("risk_level", "UNKNOWN")] += 1
    
    for risk, count in sorted(risk_counts.items()):
        print(f"  {risk}: {count} files")
    
    print("\nRecommended Migration Order:")
    print("-" * 80)
    
    for i, analysis in enumerate(migration_order, 1):
        filepath = analysis["file"]
        risk = analysis.get("risk_level", "UNKNOWN")
        score = analysis.get("risk_score", 0)
        factors = ", ".join(analysis.get("risk_factors", []))
        
        print(f"{i:3d}. [{risk:6s}] {filepath}")
        print(f"     Score: {score}, Factors: {factors or 'none'}")
        print(f"     Imports: {len(analysis.get('imports', []))}, "
              f"Instantiations: {len(analysis.get('instantiations', []))}, "
              f"Calls: {len(analysis.get('method_calls', []))}")
        print()
    
    # Save detailed report
    report = {
        "summary": {
            "total_files": len(migration_order),
            "risk_distribution": dict(risk_counts),
            "root_directory": str(root_dir)
        },
        "files": migration_order
    }
    
    with open("lmmodule_migration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: lmmodule_migration_report.json")
    
    # Generate actionable checklist
    print("\nMigration Checklist:")
    print("-" * 80)
    
    for risk_level in ["LOW", "MEDIUM", "HIGH"]:
        files = [f for f in migration_order if f.get("risk_level") == risk_level]
        if files:
            print(f"\n{risk_level} Risk Files ({len(files)} files):")
            for f in files[:5]:  # Show first 5
                print(f"  [ ] {f['file']}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")


if __name__ == "__main__":
    main()