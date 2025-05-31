"""Simple AI code reviewer using the new operator system.

This example shows how to build a practical code review tool.
You'll learn:
- How to compose operators for complex tasks
- How to structure multi-step LLM workflows
- How to produce actionable output

Requirements:
- ember
- Models: gpt-4 recommended

Example usage:
    python simple_reviewer.py
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from ember.api import models
from ember.core.module_v2 import EmberModule, Chain


class Severity(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"


@dataclass
class CodeIssue:
    """A single code issue found by the reviewer."""
    line_number: Optional[int]
    severity: Severity
    message: str
    suggestion: Optional[str]


@dataclass
class ReviewResult:
    """Complete review result."""
    summary: str
    issues: List[CodeIssue]
    score: int  # 0-100


# Sample code to review
SAMPLE_CODE = '''
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

def process_data(data):
    # Process the data
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

class UserManager:
    def __init__(self):
        self.users = {}
    
    def add_user(self, name, email):
        self.users[name] = email
        
    def get_user(self, name):
        return self.users[name]
'''


class CodeAnalyzer(EmberModule):
    """Analyze code for potential issues."""
    model: models.ModelBinding
    
    def __call__(self, code: str) -> List[CodeIssue]:
        prompt = f"""
Analyze this Python code for issues. For each issue found, provide:
LINE: <line number or 'multiple'>
SEVERITY: <info|warning|error>
MESSAGE: <brief description>
SUGGESTION: <how to fix>

Code to analyze:
```python
{code}
```

Format each issue on separate lines with the labels above.
"""
        response = self.model(prompt)
        
        # Parse response into issues
        issues = []
        current_issue = {}
        
        for line in response.text.strip().split('\n'):
            if line.startswith('LINE:'):
                if current_issue:
                    issues.append(self._create_issue(current_issue))
                current_issue = {'line': line.split(':', 1)[1].strip()}
            elif line.startswith('SEVERITY:'):
                current_issue['severity'] = line.split(':', 1)[1].strip()
            elif line.startswith('MESSAGE:'):
                current_issue['message'] = line.split(':', 1)[1].strip()
            elif line.startswith('SUGGESTION:'):
                current_issue['suggestion'] = line.split(':', 1)[1].strip()
        
        if current_issue:
            issues.append(self._create_issue(current_issue))
        
        return issues
    
    def _create_issue(self, data: dict) -> CodeIssue:
        """Create CodeIssue from parsed data."""
        line_str = data.get('line', 'unknown')
        try:
            line_num = int(line_str) if line_str != 'multiple' else None
        except:
            line_num = None
            
        severity_map = {
            'info': Severity.INFO,
            'warning': Severity.WARNING,
            'error': Severity.ERROR
        }
        severity = severity_map.get(
            data.get('severity', 'info').lower(), 
            Severity.INFO
        )
        
        return CodeIssue(
            line_number=line_num,
            severity=severity,
            message=data.get('message', 'No message'),
            suggestion=data.get('suggestion')
        )


class CodeScorer(EmberModule):
    """Score code quality from 0-100."""
    model: models.ModelBinding
    
    def __call__(self, code: str, issues: List[CodeIssue]) -> int:
        # Simple scoring based on issues
        score = 100
        for issue in issues:
            if issue.severity == Severity.ERROR:
                score -= 15
            elif issue.severity == Severity.WARNING:
                score -= 10
            else:
                score -= 5
        
        return max(0, score)


class ReviewSummarizer(EmberModule):
    """Create human-readable review summary."""
    model: models.ModelBinding
    
    def __call__(self, code: str, issues: List[CodeIssue], score: int) -> str:
        if not issues:
            return f"Code looks good! No issues found. Score: {score}/100"
        
        issue_summary = "\n".join([
            f"- {issue.severity.value.upper()}: {issue.message}"
            for issue in issues[:5]  # Top 5 issues
        ])
        
        prompt = f"""
Write a brief, constructive code review summary.
Score: {score}/100
Top issues:
{issue_summary}

Keep it encouraging and actionable in 2-3 sentences.
"""
        response = self.model(prompt)
        return response.text.strip()


class SimpleCodeReviewer(EmberModule):
    """Complete code review pipeline."""
    analyzer: CodeAnalyzer
    scorer: CodeScorer
    summarizer: ReviewSummarizer
    
    def __call__(self, code: str) -> ReviewResult:
        # Step 1: Analyze code
        issues = self.analyzer(code)
        
        # Step 2: Calculate score
        score = self.scorer(code, issues)
        
        # Step 3: Generate summary
        summary = self.summarizer(code, issues, score)
        
        return ReviewResult(
            summary=summary,
            issues=issues,
            score=score
        )


def main():
    # Setup model
    model = models.instance("gpt-4", temperature=0.3)
    
    # Create reviewer pipeline
    reviewer = SimpleCodeReviewer(
        analyzer=CodeAnalyzer(model=model),
        scorer=CodeScorer(model=model),
        summarizer=ReviewSummarizer(model=model)
    )
    
    # Review the sample code
    print("🔍 Reviewing code...\n")
    result = reviewer(SAMPLE_CODE)
    
    # Display results
    print(f"📊 Score: {result.score}/100")
    print(f"\n📝 Summary:\n{result.summary}")
    
    if result.issues:
        print(f"\n🚨 Issues found ({len(result.issues)}):")
        for i, issue in enumerate(result.issues, 1):
            line_info = f"Line {issue.line_number}" if issue.line_number else "Multiple lines"
            print(f"\n{i}. [{issue.severity.value.upper()}] {line_info}")
            print(f"   {issue.message}")
            if issue.suggestion:
                print(f"   💡 {issue.suggestion}")
    else:
        print("\n✅ No issues found!")
    
    # Show how to use individual components
    print("\n" + "="*50)
    print("You can also use components individually:")
    
    # Just analyze
    analyzer = CodeAnalyzer(model=model)
    quick_issues = analyzer("def bad_function(): pass")
    print(f"\nQuick analysis found {len(quick_issues)} issues")


if __name__ == "__main__":
    main()


# Next steps:
# 1. Add support for different languages
# 2. Integrate with GitHub PRs
# 3. Add caching for repeated reviews
# 4. Create specialized reviewers (security, performance, style)
# 5. Add interactive fix suggestions

# See also:
# - examples/applications/test_generator/ - Generate tests for code
# - examples/patterns/self_critique.py - Have AI critique its own output
# - examples/performance/caching_strategies.py - Cache review results