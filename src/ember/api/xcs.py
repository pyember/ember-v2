"""XCS API - Smart execution made simple.

The new XCS provides automatic optimization with zero configuration.
Just use @jit and let XCS handle the rest.

Examples:
    from ember.api.xcs import jit
    
    @jit
    def process(data):
        return model(data)
    
    # That's it! Automatic parallelization, caching, and optimization.

For the 10% who need more control:
    from ember.api.xcs import jit, Config
    
    @jit(config=Config(cache=False))
    def process_sensitive(data):
        return secure_model(data)
"""

# Re-export the simple API
from ember.xcs import jit, get_jit_stats
from ember.xcs.transformations import vmap

# Config for advanced users (hidden by default)
# Users must explicitly import this
from ember.xcs.config import Config as _Config

# Make Config available but not in __all__
Config = _Config

__all__ = [
    # The 90% API - core functions
    'jit',
    'get_jit_stats',
    'vmap',
    # Config is available but not advertised in __all__
]