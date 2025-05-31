"""Function introspection for natural API support.

This module analyzes Python functions to understand their calling conventions,
enabling transparent adaptation between natural Python and internal representations.
"""

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, get_type_hints


class CallStyle(Enum):
    """Function calling conventions."""
    NATURAL = "natural"              # def f(x, y, z=1)
    OPERATOR = "operator"            # def forward(self, *, inputs)
    KEYWORD_ONLY = "keyword_only"    # def f(*, x, y)
    MIXED = "mixed"                  # def f(x, *, y)
    NO_ARGS = "no_args"             # def f()


class ParameterKind(Enum):
    """Enhanced parameter classification."""
    POSITIONAL = "positional"
    KEYWORD = "keyword"
    VAR_POSITIONAL = "var_positional"    # *args
    VAR_KEYWORD = "var_keyword"          # **kwargs
    INPUTS_DICT = "inputs_dict"          # Special *, inputs parameter


@dataclass
class ParameterInfo:
    """Detailed information about a function parameter."""
    name: str
    kind: ParameterKind
    type_hint: Optional[type] = None
    default: Any = inspect.Parameter.empty
    position: Optional[int] = None
    
    @property
    def has_default(self) -> bool:
        """Check if parameter has a default value."""
        return self.default is not inspect.Parameter.empty
        
    @property
    def is_required(self) -> bool:
        """Check if parameter is required."""
        return not self.has_default


@dataclass
class FunctionMetadata:
    """Complete metadata about a function's signature and behavior."""
    func: Callable
    name: str
    call_style: CallStyle
    parameters: List[ParameterInfo]
    return_type: Optional[type]
    is_method: bool
    is_coroutine: bool
    has_var_args: bool
    has_var_kwargs: bool
    
    @property
    def positional_params(self) -> List[ParameterInfo]:
        """Get all positional parameters."""
        return [p for p in self.parameters if p.kind == ParameterKind.POSITIONAL]
    
    @property
    def keyword_params(self) -> List[ParameterInfo]:
        """Get all keyword-only parameters."""
        return [p for p in self.parameters if p.kind == ParameterKind.KEYWORD]
    
    @property
    def required_params(self) -> List[ParameterInfo]:
        """Get all required parameters."""
        return [p for p in self.parameters if p.is_required]
    
    def get_param_by_name(self, name: str) -> Optional[ParameterInfo]:
        """Get parameter info by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None


class FunctionIntrospector:
    """Analyzes function signatures and calling patterns."""
    
    def analyze(self, func: Callable) -> FunctionMetadata:
        """Deep analysis of function signature and patterns."""
        # Get base function (unwrap decorators if needed)
        base_func = inspect.unwrap(func)
        
        # Get signature and type hints
        sig = inspect.signature(base_func)
        type_hints = get_type_hints(base_func, include_extras=True)
        
        # Analyze parameters
        parameters = self._analyze_parameters(sig, type_hints)
        
        # Detect calling style
        call_style = self._detect_call_style(parameters)
        
        # Check special properties
        is_method = self._is_method(base_func)
        is_coroutine = inspect.iscoroutinefunction(base_func)
        
        # Check for var args/kwargs
        has_var_args = any(p.kind == ParameterKind.VAR_POSITIONAL for p in parameters)
        has_var_kwargs = any(p.kind == ParameterKind.VAR_KEYWORD for p in parameters)
        
        return FunctionMetadata(
            func=func,
            name=func.__name__,
            call_style=call_style,
            parameters=parameters,
            return_type=type_hints.get('return'),
            is_method=is_method,
            is_coroutine=is_coroutine,
            has_var_args=has_var_args,
            has_var_kwargs=has_var_kwargs
        )
    
    def _analyze_parameters(self, sig: inspect.Signature, 
                          type_hints: Dict[str, type]) -> List[ParameterInfo]:
        """Analyze all parameters in detail."""
        parameters = []
        position = 0
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'cls' for methods
            if param_name in ('self', 'cls'):
                continue
                
            # Determine parameter kind
            kind = self._classify_parameter(param)
            
            # Get type hint
            type_hint = type_hints.get(param_name)
            
            # Create parameter info
            param_info = ParameterInfo(
                name=param_name,
                kind=kind,
                type_hint=type_hint,
                default=param.default,
                position=position if kind == ParameterKind.POSITIONAL else None
            )
            
            parameters.append(param_info)
            if kind == ParameterKind.POSITIONAL:
                position += 1
                
        return parameters
    
    def _classify_parameter(self, param: inspect.Parameter) -> ParameterKind:
        """Classify a parameter into our enhanced categories."""
        # Special case: *, inputs parameter (Ember pattern)
        if (param.kind == inspect.Parameter.KEYWORD_ONLY and 
            param.name == 'inputs'):
            return ParameterKind.INPUTS_DICT
            
        # Standard classifications
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return ParameterKind.VAR_POSITIONAL
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            return ParameterKind.VAR_KEYWORD
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            return ParameterKind.KEYWORD
        else:
            return ParameterKind.POSITIONAL
    
    def _detect_call_style(self, parameters: List[ParameterInfo]) -> CallStyle:
        """Detect the calling convention of a function."""
        if not parameters:
            return CallStyle.NO_ARGS
            
        # Check for operator style: single *, inputs parameter
        if (len(parameters) == 1 and 
            parameters[0].kind == ParameterKind.INPUTS_DICT):
            return CallStyle.OPERATOR
            
        # Check if all parameters are keyword-only
        non_var_params = [p for p in parameters 
                         if p.kind not in (ParameterKind.VAR_POSITIONAL, 
                                         ParameterKind.VAR_KEYWORD)]
        
        if all(p.kind == ParameterKind.KEYWORD for p in non_var_params):
            return CallStyle.KEYWORD_ONLY
            
        # Check if all parameters are positional
        if all(p.kind == ParameterKind.POSITIONAL for p in non_var_params):
            return CallStyle.NATURAL
            
        # Mixed style
        return CallStyle.MIXED
    
    def _is_method(self, func: Callable) -> bool:
        """Check if function is a method."""
        # Check if function has __self__ attribute (bound method)
        if hasattr(func, '__self__'):
            return True
            
        # Check if first parameter is 'self' or 'cls'
        try:
            sig = inspect.signature(func)
            first_param = next(iter(sig.parameters.keys()), None)
            return first_param in ('self', 'cls')
        except:
            return False
    
    def detect_style(self, func: Callable) -> CallStyle:
        """Quick detection of calling style without full analysis."""
        metadata = self.analyze(func)
        return metadata.call_style
    
    def extract_type_hints(self, func: Callable) -> Dict[str, type]:
        """Extract and preserve type information."""
        return get_type_hints(func, include_extras=True)
    
    def is_compatible_with_style(self, func: Callable, style: CallStyle) -> bool:
        """Check if function is compatible with a given calling style."""
        current_style = self.detect_style(func)
        
        # Exact match
        if current_style == style:
            return True
            
        # Natural functions can be adapted to any style
        if current_style == CallStyle.NATURAL:
            return True
            
        # No-arg functions are compatible with most styles
        if current_style == CallStyle.NO_ARGS:
            return style != CallStyle.OPERATOR
            
        return False