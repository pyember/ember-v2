# Data Metadata: The Masters' Convergence

*What happens when Dean, Ghemawat, Brockman, Martin, Jobs, Ritchie, Knuth, and Carmack design metadata together*

## The Thought Experiment

Imagine these masters sitting together, designing a metadata system for datasets. Each brings their perspective:

### Initial Positions

**Knuth**: "We need complete statistical characterization - means, variances, distributions, theoretical properties..."

**Carmack**: "Just tell me how fast it loads and how much RAM it needs."

**Jobs**: "Users shouldn't even see metadata. It should just work."

**Martin**: "Every field should reveal intent. Why does this data exist?"

**Ritchie**: "struct dataset_info { size_t size; char* format; char* source; };"

**Dean & Ghemawat**: "We need hints for the optimizer - access patterns, parallelization opportunities..."

**Brockman**: "Show me one real example so I know what I'm working with."

## The Convergence Process

### Round 1: What's Essential?

**Carmack**: "Look, 90% of problems come from not knowing size and speed. Start there."

**Everyone nods**

**Ritchie**: "Agreed. Size, format, source. What else is truly needed?"

### Round 2: Developer Experience

**Brockman**: "People waste hours figuring out data shape. One example prevents that."

**Jobs**: "One *good* example. Not twenty fields of maybe-useful information."

**Martin**: "The example IS documentation. It reveals intent better than description fields."

### Round 3: Performance

**Dean**: "We need batch size hints. Bad batching kills performance."

**Ghemawat**: "But computed, not configured. Measure and recommend."

**Carmack**: "Yes. Tell me the optimal batch size you discovered, not what some config says."

### Round 4: Simplification

**Ritchie**: "We're overthinking. What breaks if we remove each field?"

**Jobs**: "Remove fields until it stops working, then add back one."

**Knuth**: "I concede. Statistical properties belong in analysis tools, not core metadata."

## Their Final Design

```python
@dataclass
class DatasetMetadata:
    """Only what you need to use the dataset successfully."""
    
    # Core identity (Ritchie: the minimum)
    name: str
    source: str
    
    # Performance reality (Carmack: what actually matters)
    size_bytes: int
    estimated_examples: int
    typical_load_time_ms: float
    
    # Optimization hints (Dean & Ghemawat: measured, not guessed)
    recommended_batch_size: int  # Based on actual benchmarks
    
    # Developer success (Brockman: one example is worth 1000 words)
    example_item: Dict[str, Any]
    
    # Intent (Martin: why does this exist?)
    task_type: Literal["classification", "generation", "ranking", "other"]
    description: str  # One sentence, not an essay
```

## What They Explicitly Rejected

**Complex Validation Schemas**: "Validation belongs in user code, not framework metadata." - Ritchie

**Statistical Properties**: "Compute them if needed, don't store them." - Carmack

**Extensive Type Information**: "The example shows types better than schemas." - Brockman

**Configuration Options**: "Measure optimal values, don't make users guess." - Dean

**Feature Catalogs**: "If you need a catalog, your data is too complex." - Jobs

## The Wisdom

**Jobs**: "Perfection is achieved when there is nothing left to take away."

**Carmack**: "Every field should earn its keep by preventing real errors."

**Martin**: "Metadata should make the right thing obvious and the wrong thing hard."

**Ritchie**: "When in doubt, leave it out."

## Implementation Implications

1. **Compute, Don't Store**: Derived properties should be computed when needed
2. **Measure, Don't Configure**: Performance hints from benchmarks, not settings
3. **Example Over Schema**: One real example beats complex type definitions
4. **Essential Only**: Each field must prevent actual user errors

## The Test

Before adding any metadata field, ask:
1. What error does this prevent?
2. Can it be computed instead?
3. Does the example already show this?
4. Would Ritchie include it in a C struct?

If any answer is unclear, the field doesn't belong.

## Day 0 Implementation

```python
# src/ember/data/metadata.py
from dataclasses import dataclass
from typing import Dict, Any, Literal

@dataclass(slots=True)  # Ritchie would approve
class DatasetMetadata:
    """Only what prevents errors."""
    # Identity
    name: str
    source: str
    
    # Performance  
    size_bytes: int
    estimated_examples: int
    typical_load_time_ms: float
    
    # Optimization
    recommended_batch_size: int
    
    # Success
    example_item: Dict[str, Any]
    
    # Intent
    task_type: Literal["classification", "generation", "ranking", "other"]
    description: str  # One sentence max

# Day 0: Replace DatasetInfo with this everywhere
# Day 0: Remove DataItem normalizer (>150 LOC → <50 LOC)
```