"""Selector functions for choosing from multiple options.

Provides simple selection strategies for ensemble outputs.
"""

from collections import Counter
from typing import List, Any, Optional, Callable


def most_common(items: List[Any]) -> Any:
    """Return the most common item from a list.
    
    Args:
        items: List of items to select from
        
    Returns:
        Most common item
    """
    if not items:
        return None
    counter = Counter(items)
    return counter.most_common(1)[0][0]


def best_of(items: List[Any], scorer: Optional[Callable] = None) -> Any:
    """Return the best item according to a scoring function.
    
    Args:
        items: List of items to select from
        scorer: Optional scoring function
        
    Returns:
        Best item according to scorer
    """
    if not items:
        return None
    if scorer is None:
        # Default: return first non-None item
        for item in items:
            if item is not None:
                return item
        return items[0]
    return max(items, key=scorer)


class MajorityVote:
    """Majority vote selector with configurable threshold."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def __call__(self, items: List[Any]) -> Any:
        """Select by majority vote.
        
        Args:
            items: List of items to vote on
            
        Returns:
            Most common item if it exceeds threshold, else first item
        """
        if not items:
            return None
            
        counter = Counter(items)
        most_common_item, count = counter.most_common(1)[0]
        
        if count / len(items) >= self.threshold:
            return most_common_item
        return items[0]  # No clear majority, return first