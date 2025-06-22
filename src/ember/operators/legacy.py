"""Legacy operator compatibility layer.

Provides backward compatibility for old EmberModule-based operators.
This allows gradual migration while maintaining existing functionality.
"""

from typing import Any, Dict, Optional

from ember.core.registry.operator.base._module import EmberModule
from ember.core.registry.operator.base.operator_base import Operator as OldOperator

__all__ = [
    'EmberModule',
    'LegacyOperator',
    'modernize',
]


# Re-export EmberModule for compatibility
LegacyOperator = OldOperator


def modernize(legacy_operator: EmberModule) -> Any:
    """Convert a legacy operator to modern style.
    
    This provides a migration path from old EmberModule-based operators
    to the new simplified operator system.
    
    Args:
        legacy_operator: Legacy EmberModule instance or class
        
    Returns:
        Modern operator that maintains the same functionality
        
    Example:
        >>> # Old style
        >>> class OldOp(EmberModule):
        ...     def forward(self, inputs):
        ...         return {"result": inputs["text"].upper()}
        >>> 
        >>> # Convert to modern
        >>> modern_op = modernize(OldOp())
        >>> result = modern_op({"text": "hello"})
    """
    from ember.operators.advanced import Operator, operator
    
    # Handle class vs instance
    if isinstance(legacy_operator, type):
        # It's a class, instantiate it first
        instance = legacy_operator()
    else:
        # It's already an instance
        instance = legacy_operator
    
    @operator.advanced
    class ModernizedOperator(Operator):
        """Modernized version of legacy operator."""
        
        def __init__(self, legacy_instance):
            """Initialize with legacy instance."""
            super().__init__()
            self._legacy = legacy_instance
            
            # Copy relevant attributes
            for attr in dir(legacy_instance):
                if not attr.startswith('_') and not callable(getattr(legacy_instance, attr)):
                    setattr(self, attr, getattr(legacy_instance, attr))
        
        def __call__(self, *args, **kwargs):
            """Execute using legacy forward method."""
            # Legacy operators expect 'inputs' in kwargs
            if args and not kwargs.get('inputs'):
                kwargs['inputs'] = args[0]
            
            # Call legacy forward method
            if hasattr(self._legacy, 'forward'):
                return self._legacy.forward(**kwargs)
            elif hasattr(self._legacy, '__call__'):
                return self._legacy(**kwargs)
            else:
                raise RuntimeError("Legacy operator has no forward or __call__ method")
        
        def tree_flatten(self):
            """Delegate to legacy tree methods if available."""
            if hasattr(self._legacy, '_tree_flatten'):
                return self._legacy._tree_flatten()
            else:
                # Default implementation
                return [], {'legacy': self._legacy}
        
        @classmethod
        def tree_unflatten(cls, aux_data, values):
            """Reconstruct from tree representation."""
            return cls(aux_data['legacy'])
    
    # Create and return modernized instance
    return ModernizedOperator(instance)


class MigrationHelper:
    """Helper class for migrating legacy operators."""
    
    @staticmethod
    def analyze_legacy_operator(operator_class: type) -> Dict[str, Any]:
        """Analyze a legacy operator class for migration.
        
        Args:
            operator_class: Legacy operator class
            
        Returns:
            Dictionary with migration analysis
        """
        analysis = {
            'uses_tree_protocol': hasattr(operator_class, '_tree_flatten'),
            'uses_metadata': hasattr(operator_class, '_metadata'),
            'uses_static_fields': hasattr(operator_class, '_static_fields'),
            'uses_dynamic_fields': hasattr(operator_class, '_dynamic_fields'),
            'complexity': 'low',
            'recommended_tier': 'simple',
            'migration_notes': []
        }
        
        # Check complexity
        if analysis['uses_tree_protocol']:
            analysis['complexity'] = 'high'
            analysis['recommended_tier'] = 'advanced'
            analysis['migration_notes'].append(
                "Uses tree protocol - consider advanced.TreeProtocol"
            )
        
        if analysis['uses_static_fields'] or analysis['uses_dynamic_fields']:
            analysis['complexity'] = 'medium'
            if analysis['recommended_tier'] == 'simple':
                analysis['recommended_tier'] = 'advanced'
            analysis['migration_notes'].append(
                "Uses field separation - may need advanced.static_field"
            )
        
        # Check for common patterns
        source = None
        try:
            import inspect
            source = inspect.getsource(operator_class)
        except:
            pass
        
        if source:
            if 'vmap' in source or 'pmap' in source:
                analysis['migration_notes'].append(
                    "Uses XCS transformations - ensure compatibility"
                )
            
            if 'EmberContext' in source:
                analysis['migration_notes'].append(
                    "Uses EmberContext - may need refactoring"
                )
        
        return analysis
    
    @staticmethod
    def generate_migration_code(operator_class: type) -> str:
        """Generate migration code for a legacy operator.
        
        Args:
            operator_class: Legacy operator class
            
        Returns:
            String with suggested migration code
        """
        analysis = MigrationHelper.analyze_legacy_operator(operator_class)
        
        if analysis['recommended_tier'] == 'simple':
            return f"""# Simple function replacement
def {operator_class.__name__.lower()}(inputs):
    \"\"\"Migrated from {operator_class.__name__}.\"\"\"
    # TODO: Implement logic from {operator_class.__name__}.forward()
    return {{"result": "TODO"}}
"""
        
        elif analysis['recommended_tier'] == 'advanced':
            return f"""# Advanced operator with full features
from ember.operators.advanced import Operator, operator, static_field

@operator.advanced
class {operator_class.__name__}(Operator):
    \"\"\"Migrated from legacy {operator_class.__name__}.\"\"\"
    
    # TODO: Add fields from original operator
    # config: Dict = static_field(default_factory=dict)
    
    def __call__(self, inputs):
        \"\"\"Execute operator logic.\"\"\"
        # TODO: Implement logic from forward() method
        return {{"result": "TODO"}}
    
    def tree_flatten(self):
        \"\"\"Support tree transformations.\"\"\"
        # TODO: Implement based on original _tree_flatten
        return [], {{"config": self.config}}
    
    @classmethod
    def tree_unflatten(cls, aux_data, values):
        \"\"\"Reconstruct from tree representation.\"\"\"
        return cls(**aux_data)
"""
        
        else:
            return "# Complex operator - manual migration recommended"