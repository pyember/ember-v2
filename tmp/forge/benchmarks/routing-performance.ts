/**
 * Performance benchmarks for routing decisions
 */

import { ProviderRouter } from '../src/core/ember-bridge';

interface BenchmarkResult {
  operation: string;
  iterations: number;
  totalTime: number;
  averageTime: number;
  opsPerSecond: number;
}

function benchmark(
  name: string,
  fn: () => void,
  iterations: number = 100000
): BenchmarkResult {
  const start = process.hrtime.bigint();
  
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  
  const end = process.hrtime.bigint();
  const totalTime = Number(end - start) / 1_000_000; // Convert to ms
  const averageTime = totalTime / iterations;
  const opsPerSecond = 1000 / averageTime;
  
  return {
    operation: name,
    iterations,
    totalTime,
    averageTime,
    opsPerSecond,
  };
}

function runBenchmarks() {
  console.log('🔨 Forge Routing Performance Benchmarks\n');
  
  const router = new ProviderRouter();
  const results: BenchmarkResult[] = [];
  
  // Test data
  const toolMessage = [{ role: 'user' as const, content: 'List all files' }];
  const planningMessage = [{ role: 'user' as const, content: 'How should I approach this refactor?' }];
  const codeMessage = [{ role: 'user' as const, content: 'Implement a binary search tree' }];
  const tools = [{ type: 'function' as const, function: { name: 'ls' } }];
  
  // Benchmark intent detection
  results.push(
    benchmark('Intent Detection - Tool Use', () => {
      router.detectIntent(toolMessage, tools);
    })
  );
  
  results.push(
    benchmark('Intent Detection - Planning', () => {
      router.detectIntent(planningMessage);
    })
  );
  
  results.push(
    benchmark('Intent Detection - Code Gen', () => {
      router.detectIntent(codeMessage);
    })
  );
  
  // Benchmark provider lookup
  results.push(
    benchmark('Provider Lookup - Known Intent', () => {
      router.getProvider('tool_use');
    })
  );
  
  results.push(
    benchmark('Provider Lookup - Default', () => {
      router.getProvider('unknown_intent');
    })
  );
  
  // Print results
  console.table(results.map(r => ({
    Operation: r.operation,
    'Avg Time (μs)': (r.averageTime * 1000).toFixed(3),
    'Ops/Second': r.opsPerSecond.toLocaleString('en-US', { maximumFractionDigits: 0 }),
  })));
  
  // Performance assertions
  console.log('\n📊 Performance Analysis:');
  
  const slowest = results.reduce((prev, curr) => 
    curr.averageTime > prev.averageTime ? curr : prev
  );
  
  const fastest = results.reduce((prev, curr) => 
    curr.averageTime < prev.averageTime ? curr : prev
  );
  
  console.log(`Fastest: ${fastest.operation} (${(fastest.averageTime * 1000).toFixed(3)}μs)`);
  console.log(`Slowest: ${slowest.operation} (${(slowest.averageTime * 1000).toFixed(3)}μs)`);
  
  // Check if routing is fast enough (< 1ms average)
  const allFastEnough = results.every(r => r.averageTime < 1);
  if (allFastEnough) {
    console.log('✅ All operations complete in under 1ms');
  } else {
    console.warn('⚠️  Some operations exceed 1ms threshold');
  }
  
  // Memory usage
  const memUsage = process.memoryUsage();
  console.log('\n💾 Memory Usage:');
  console.log(`Heap Used: ${(memUsage.heapUsed / 1024 / 1024).toFixed(2)} MB`);
  console.log(`RSS: ${(memUsage.rss / 1024 / 1024).toFixed(2)} MB`);
}

// Run benchmarks
if (require.main === module) {
  runBenchmarks();
}