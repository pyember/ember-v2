"""Python execution tracer using sys.settrace.

This module implements runtime tracing of Python functions to build execution graphs.
No AST analysis, no magic - just recording what actually happens.
"""

import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Tuple


@dataclass
class Operation:
    """Recorded operation during tracing."""

    func: Callable
    args: tuple
    kwargs: dict
    result: Any
    operation_id: int
    dependencies: Set[int] = field(default_factory=set)

    def __hash__(self):
        return hash(self.operation_id)


class TracingError(Exception):
    """Raised when tracing fails."""

    pass


class PythonTracer:
    """Traces Python execution using sys.settrace.

    Records all function calls and their dependencies to build an execution graph.
    Thread-safe and handles nested calls correctly.
    """

    def __init__(self):
        self.operations: List[Operation] = []
        self.tracing = False
        self.target_frame = None
        self.operation_counter = 0
        self.result_to_operation: Dict[int, int] = {}  # object id -> operation id
        self.current_dependencies: Set[int] = set()
        self._lock = threading.Lock()

    def trace_function(self, func: Callable, args: tuple, kwargs: dict) -> List[Operation]:
        """Trace function execution and return operations.

        Args:
            func: Function to trace
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            List of operations recorded during execution

        Raises:
            TracingError: If tracing fails
        """
        with self._lock:
            if self.tracing:
                raise TracingError("Already tracing another function")

            self.operations = []
            self.operation_counter = 0
            self.result_to_operation = {}
            self.current_dependencies = set()
            self.tracing = True

        # Set up tracing
        old_trace = sys.gettrace()
        sys.settrace(self._trace_calls)

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Record the main function as an operation
            main_op = Operation(
                func=func,
                args=args,
                kwargs=kwargs,
                result=result,
                operation_id=self.operation_counter,
                dependencies=set(range(self.operation_counter)),
            )
            self.operations.append(main_op)

            return self.operations
        except Exception as e:
            raise TracingError(f"Function execution failed: {e}") from e
        finally:
            sys.settrace(old_trace)
            with self._lock:
                self.tracing = False
                self.target_frame = None

    def _trace_calls(self, frame, event: str, arg: Any):
        """Trace callback for sys.settrace.

        Records function calls and returns within the target function.
        """
        if not self.tracing:
            return None

        # Set target frame on first call
        if self.target_frame is None and event == "call":
            self.target_frame = frame

        # Only trace within our target function
        if not self._is_in_target_frame(frame):
            return None

        if event == "call":
            # Skip tracer code itself
            if self._is_tracer_code(frame):
                return None

            # Record dependencies based on arguments
            arg_deps = self._extract_dependencies_from_args(frame)
            self.current_dependencies = arg_deps

        elif event == "return":
            # Skip returns from frames we didn't enter
            if frame is self.target_frame:
                # Don't record the target frame return here - we do it in trace_function
                return self._trace_calls

            # Record the operation
            func_name = frame.f_code.co_name

            # Skip certain internal functions
            if func_name.startswith("_") and func_name not in {"__init__", "__call__"}:
                return self._trace_calls

            if func_name not in {"<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"}:
                # Regular function return
                self._record_operation_from_frame(frame, arg)
            else:
                # Handle comprehensions specially
                self._handle_comprehension_return(frame, arg)

        return self._trace_calls

    def _is_in_target_frame(self, frame) -> bool:
        """Check if frame is within our target function."""
        current = frame
        while current is not None:
            if current is self.target_frame:
                return True
            current = current.f_back
        return False

    def _is_tracer_code(self, frame) -> bool:
        """Check if frame is from tracer code itself."""
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        # Only filter out the tracer's own methods
        return filename == __file__ and func_name in {
            "trace_function",
            "_trace_calls",
            "_is_in_target_frame",
            "_is_tracer_code",
            "_extract_dependencies_from_args",
            "_record_operation_from_frame",
            "_extract_args_from_frame",
            "_handle_comprehension_return",
        }

    def _extract_dependencies_from_args(self, frame) -> Set[int]:
        """Extract operation dependencies from function arguments."""
        deps = set()

        # Check local variables for results of previous operations
        for var_name, var_value in frame.f_locals.items():
            if var_name == "self":  # Skip self in methods
                continue

            obj_id = id(var_value)
            if obj_id in self.result_to_operation:
                deps.add(self.result_to_operation[obj_id])

        return deps

    def _record_operation_from_frame(self, frame, result):
        """Record an operation from a frame."""
        # Extract function object
        func_name = frame.f_code.co_name

        # Try multiple ways to get the function object
        func = None

        # 1. Check globals (for regular functions)
        func = frame.f_globals.get(func_name)

        # 2. Check if it's a method on self
        if func is None:
            self_obj = frame.f_locals.get("self")
            if self_obj:
                func = getattr(self_obj, func_name, None)

        # 3. Check builtins
        if func is None:
            import builtins

            func = getattr(builtins, func_name, None)

        # 4. For lambdas and nested functions, try to find in locals
        if func is None and func_name == "<lambda>":
            # For lambdas, we can't easily get the function object
            # Just use the code object as a placeholder
            func = frame.f_code

        if func is None:
            # Can't find the function - skip
            return

        # Extract arguments
        args, kwargs = self._extract_args_from_frame(frame)

        # Create operation
        op = Operation(
            func=func,
            args=args,
            kwargs=kwargs,
            result=result,
            operation_id=self.operation_counter,
            dependencies=self.current_dependencies.copy(),
        )

        self.operations.append(op)

        # Map result to operation for dependency tracking
        if result is not None:
            self.result_to_operation[id(result)] = self.operation_counter

        self.operation_counter += 1

    def _extract_args_from_frame(self, frame) -> Tuple[tuple, dict]:
        """Extract function arguments from frame."""
        code = frame.f_code
        varnames = code.co_varnames
        argcount = code.co_argcount
        kwonlycount = code.co_kwonlyargcount

        # Extract positional arguments
        args = []
        for i in range(argcount):
            if varnames[i] != "self":  # Skip self
                args.append(frame.f_locals.get(varnames[i]))

        # Extract keyword-only arguments
        kwargs = {}
        for i in range(argcount, argcount + kwonlycount):
            var_name = varnames[i]
            if var_name in frame.f_locals:
                kwargs[var_name] = frame.f_locals[var_name]

        return tuple(args), kwargs

    def _handle_comprehension_return(self, frame, result):
        """Handle list/dict/set comprehension returns."""
        # Comprehensions are special - they call a function for each item
        # We need to record these as a batch operation

        # Find the comprehension expression in parent frame
        parent_frame = frame.f_back
        if parent_frame and isinstance(result, (list, dict, set)):
            # Record as a single batch operation
            op = Operation(
                func=type(result),  # list, dict, or set constructor
                args=(result,),
                kwargs={},
                result=result,
                operation_id=self.operation_counter,
                dependencies=self.current_dependencies.copy(),
            )
            self.operations.append(op)

            if result:
                self.result_to_operation[id(result)] = self.operation_counter

            self.operation_counter += 1


def is_traceable(func: Callable) -> bool:
    """Check if a function can be traced.

    Some functions cannot be traced:
    - Built-in functions (implemented in C)
    - Async functions
    - Generators (need special handling)
    """
    if isinstance(func, types.BuiltinFunctionType):
        return False

    if isinstance(func, types.CoroutineType):
        return False

    # Check for async function
    import inspect

    if inspect.iscoroutinefunction(func):
        return False

    # Try to get source - if we can't, probably can't trace
    try:
        inspect.getsource(func)
        return True
    except (OSError, TypeError):
        return False
