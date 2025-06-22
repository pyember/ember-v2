# JIT Strategy Selection: Visual Comparison

## Current Design vs New Design

### Current: Monolithic Scoring
```
StructuralStrategy.analyze()
│
├─ Check if class (+20) ──────────┐
├─ Check forward() (+30) ─────────┤ 
├─ Check nested ops (+40) ────────┤──> Magic Score: 0-100
├─ Check specification (+10) ─────┤
└─ Return score + rationale ──────┘

TraceStrategy.analyze()  
│
├─ Count lines (<20: +30) ────────┐
├─ Check control flow (+20) ──────┤──> Magic Score: 0-55  
├─ Base score (+5) ───────────────┤
└─ Return score + rationale ──────┘

Selection: Highest score wins (100 > 55 → Structural)
```

### New: Clean Separation of Concerns
```
Target (Class/Function)
    │
    v
FeatureDetector.detect()
    │
    ├─ IS_CLASS? ────────────┐
    ├─ HAS_FORWARD? ─────────┤
    ├─ HAS_NESTED_OPS? ──────┤──> FeatureSet {features, metadata}
    ├─ IS_SIMPLE? ───────────┤
    └─ ... ──────────────────┘
            │
            v
    StrategySelector.select()
            │
            ├─ Match against rules:
            │   
            │   Rule 1: Structural
            │   - Required: [IS_CLASS, HAS_FORWARD]
            │   - Preferred: [HAS_NESTED_OPS]
            │   - Forbidden: [CREATES_OPS_DYNAMICALLY]
            │   - Priority: 100
            │   
            │   Rule 2: Trace  
            │   - Required: [IS_FUNCTION]
            │   - Preferred: [IS_SIMPLE]
            │   - Priority: 50
            │
            └─> Best matching rule → Strategy
```

## Why This Is Better

### 1. No Magic Numbers
**Before:**
```python
score += 40  # Why 40? What does this mean?
```

**After:**
```python
Feature.HAS_NESTED_OPERATORS  # Self-documenting
```

### 2. Extensible Without Code Changes
**Before:** Must modify strategy classes
```python
class StructuralStrategy:
    def analyze(self, func):
        # Add new check here
        if my_new_check(func):
            score += 25  # Another magic number!
```

**After:** Just add a rule
```python
selector.rules.append(StrategyRule(
    strategy_name="my_strategy",
    required_features=frozenset([Feature.MY_FEATURE]),
    priority=150
))
```

### 3. Clear Decision Logic
**Before:** Buried in implementation
```python
# Somewhere in 100+ lines of analyze()...
if has_operator_fields:
    score += 40
    rationale.append("Has nested operator fields")
```

**After:** Declarative rules
```python
StrategyRule(
    required_features=frozenset([Feature.HAS_NESTED_OPERATORS])
)
```

### 4. Testable Components
**Before:** Test the whole strategy
```python
def test_structural_strategy():
    strategy = StructuralStrategy()
    result = strategy.analyze(MyClass)
    assert result['score'] == 90  # Brittle!
```

**After:** Test each piece
```python
def test_detects_nested_operators():
    features = FeatureDetector.detect(MyClass)
    assert features.has(Feature.HAS_NESTED_OPERATORS)

def test_rule_matches():
    rule = StrategyRule(required_features=...)
    assert rule_matches(rule, features)
```

## Performance Comparison

### Current: O(n) scoring per strategy
```
For each strategy:
    - Run full analyze() method
    - Complex scoring logic
    - String concatenation for rationale
```

### New: O(1) feature detection + O(n) rule matching
```
Once: Detect all features (single pass)
Then: Match against rules (simple set operations)
```

## Debugging Experience

### Current: Opaque Score
```
DEBUG: Strategy structural: score=90, reason=It's a class; Has 'forward' method; Has nested operator fields
DEBUG: Strategy trace: score=35, reason=Simple function (< 20 lines); Basic fallback strategy
```
What does score 90 mean? Why did structural win?

### New: Transparent Matching
```
DEBUG: Target features: {IS_CLASS, HAS_FORWARD_METHOD, HAS_NESTED_OPERATORS}
DEBUG: Rule 'structural' matches:
  - Required: ✓ [IS_CLASS, HAS_FORWARD_METHOD]  
  - Preferred: ✓ [HAS_NESTED_OPERATORS] (+10)
  - Forbidden: ✓ none present
  - Score: 110 (priority: 100 + preferred: 10)
DEBUG: Selected: structural (score: 110)
```

## Maintenance Over Time

### Current: Decay and Drift
- Magic numbers lose meaning
- Scores drift as people add "+5 here, +10 there"
- Hard to understand why decisions are made
- Fear of changing anything

### New: Stable and Clear
- Features have semantic meaning
- Rules are data, not code
- Easy to see why decisions are made
- Safe to extend

## The Bottom Line

The new design is what you'd see in:
- **Google's Borg**: Rule-based scheduling
- **Kubernetes**: Declarative configuration
- **TensorFlow**: Feature detection for optimization

It's not clever. It's not tricky. It just works.

**This is how distinguished engineers build systems that last.**