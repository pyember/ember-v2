"""XCS: Smart execution for Ember.

Just use @jit. That's it.

Examples:
    from ember.xcs import jit
    
    @jit
    def my_function(x):
        return model(x)
    
    # Automatic optimization with zero configuration!

For advanced users (rare):
    from ember.xcs import jit, vmap, pmap, scan, grad
    
    # Intelligent batching
    @vmap
    def batch_process(x):
        return model(x)
    
    # Distributed execution
    @pmap
    def distributed_process(x):
        return model(x)
    
    # Automatic differentiation
    @grad
    def loss_function(params):
        return compute_loss(params)
"""

# Core transformations - these subsume JAX transformations
from ember.xcs.transformations import jit, vmap, pmap, scan, grad

# Simple utility for stats
from ember.xcs._simple import get_jit_stats

# That's the complete public API!
__all__ = ['jit', 'vmap', 'pmap', 'scan', 'grad', 'get_jit_stats']

# Note: The old complex API is preserved but hidden.
# Power users can still access it via ember.xcs.legacy
# but we don't advertise this.
