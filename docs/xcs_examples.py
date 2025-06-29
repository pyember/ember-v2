"""XCS Usage Examples.

Beautiful, simple, powerful - as Jobs would want.
"""

from ember.xcs import jit, vmap, pmap, trace, optimize, export, get_graph, visualize


# Example 1: Basic JIT compilation
print("=== Example 1: Basic JIT ===")


@jit
def add(x, y):
    return x + y


result = add(5, 3)
print(f"add(5, 3) = {result}")

# The function is compiled on first call
# Subsequent calls use the optimized version


# Example 2: Automatic parallelism discovery
print("\n=== Example 2: Automatic Parallelism ===")


@jit
def parallel_compute(x, y, z):
    # These operations are independent and will run in parallel
    a = x * 2
    b = y + 10
    c = z**2

    # This runs after the parallel operations complete
    result = a + b + c
    return result


result = parallel_compute(5, 3, 4)
print(f"parallel_compute(5, 3, 4) = {result}")

# Visualize the parallelism
graph = get_graph(parallel_compute)
print(graph.visualize())


# Example 3: Vectorization with vmap
print("\n=== Example 3: Vectorization ===")


@vmap
def square(x):
    return x * x


numbers = [1, 2, 3, 4, 5]
squares = square(numbers)
print(f"square({numbers}) = {squares}")


# Example 4: Combining JIT and vmap
print("\n=== Example 4: JIT + vmap ===")


@jit
@vmap
def process_item(item):
    # Complex processing that benefits from both JIT and vectorization
    score = item * 2
    adjusted = score + 0.5
    final = adjusted**0.5
    return final


items = [10, 20, 30, 40, 50]
results = process_item(items)
print(f"Processed items: {results}")


# Example 5: Direct graph manipulation
print("\n=== Example 5: Direct Graph API ===")


def my_computation(x, y):
    a = x + y
    b = x * y
    c = a / b
    return c


# Trace the function to build a graph
graph = trace(my_computation, x=5, y=3)

# Optimize the graph
optimized = optimize(graph)

# Execute with different inputs
result1 = graph.execute({"x": 10, "y": 2})
result2 = optimized.execute({"x": 10, "y": 2})
print(f"Original: {result1}, Optimized: {result2}")


# Example 6: Export for remote execution
print("\n=== Example 6: Export for Remote ===")


@jit
def remote_compute(data):
    # Preprocessing
    cleaned = [d * 0.9 for d in data]

    # Main computation
    results = [c**2 + 1 for c in cleaned]

    # Aggregation
    return sum(results) / len(results)


# Compile the function
remote_compute([1, 2, 3])

# Export the compiled graph
graph = get_graph(remote_compute)
export_data = export(graph)

print(f"Exported {len(export_data['ops'])} operations")
print(f"Parallelism waves: {len(export_data['parallelism'])}")


# Example 7: Real-world ensemble model
print("\n=== Example 7: Ensemble Model ===")


@jit
def ensemble_predict(x, models, weights):
    """Run multiple models in parallel and combine results."""
    # Run each model - these execute in parallel!
    predictions = []
    for model in models:
        pred = model(x)
        predictions.append(pred)

    # Weighted average
    weighted_sum = sum(p * w for p, w in zip(predictions, weights))
    total_weight = sum(weights)

    return weighted_sum / total_weight


# Mock models
model1 = lambda x: x * 2
model2 = lambda x: x * 2.1
model3 = lambda x: x * 1.9

models = [model1, model2, model3]
weights = [0.3, 0.5, 0.2]

result = ensemble_predict(10, models, weights)
print(f"Ensemble prediction: {result}")


# Example 8: Performance comparison
print("\n=== Example 8: Performance Demo ===")

import time


def expensive_computation(x):
    """Simulate expensive parallel computations."""
    # Multiple independent expensive operations
    a = sum(x[i] * 1.0001 for i in range(len(x)))
    b = sum(x[i] * 0.9999 for i in range(len(x)))
    c = sum(x[i] * 1.0002 for i in range(len(x)))
    d = sum(x[i] * 0.9998 for i in range(len(x)))

    # Combine results
    return (a + b) * (c + d)


# JIT compile
jit_version = jit(expensive_computation)

# Test data
test_data = list(range(1000))

# Warm up JIT
jit_version(test_data)

# Time comparison
start = time.time()
result1 = expensive_computation(test_data)
regular_time = time.time() - start

start = time.time()
result2 = jit_version(test_data)
jit_time = time.time() - start

print(f"Regular time: {regular_time:.4f}s")
print(f"JIT time: {jit_time:.4f}s")
print(f"Results match: {abs(result1 - result2) < 0.0001}")


# Example 9: Building complex pipelines
print("\n=== Example 9: Complex Pipeline ===")

from ember.xcs import make_parallel


# Define pipeline stages
def stage1(x):
    return x * 2


def stage2(x):
    return x + 10


def stage3(x):
    return x**0.5


# Create parallel stage
parallel_stage = make_parallel(stage1, stage2, stage3)


# Use in pipeline
@jit
def pipeline(x):
    # Run three operations in parallel
    r1, r2, r3 = parallel_stage(x)

    # Combine results
    combined = r1 + r2 + r3

    # Final processing
    return combined / 3


result = pipeline(16)
print(f"Pipeline result: {result}")


# Example 10: Debugging and inspection
print("\n=== Example 10: Debugging ===")


@jit
def debug_me(x, y):
    a = x + 1
    b = y * 2
    c = a * b
    return c


# Execute
result = debug_me(5, 3)

# Inspect
stats = get_stats(debug_me)
print(f"Compilation stats: {stats}")

# Get detailed graph
graph = get_graph(debug_me)
print("\nGraph structure:")
print(graph.visualize())

# Export for analysis
exported = export(graph)
print(f"\nExported operations:")
for op in exported["ops"]:
    print(f"  {op['id']}: {op['func']}({', '.join(op['inputs'])}) -> {op['output']}")


print("\n=== Summary ===")
print("XCS makes parallel programming simple:")
print("1. Use @jit to compile any function")
print("2. Parallelism is discovered automatically")
print("3. Use @vmap/@pmap for explicit vectorization")
print("4. Direct graph API for advanced use cases")
print("5. Export graphs for distributed execution")
print("\nNo configuration needed - it just works!")
