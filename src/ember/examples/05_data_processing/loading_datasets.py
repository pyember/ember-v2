"""
Example: Loading Datasets - Working with Ember's Data API
Difficulty: Basic
Time: ~5 minutes
Prerequisites: 02_core_concepts/operators_basics.py

Learning Objectives:
- Use Ember's dataset registry
- Load and process datasets
- Create custom datasets
- Stream large datasets efficiently

Key Concepts:
- DataContext and registry
- Dataset builders
- Streaming vs materialization
- Data transformations
"""

import sys
from pathlib import Path
from typing import Iterator, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import data, operators


def main():
    """Learn to work with datasets in Ember."""
    print_section_header("Working with Datasets")
    
    # Part 1: Simple Data Loading
    print("📁 Part 1: Creating Simple Datasets\n")
    
    # Create a simple in-memory dataset
    simple_data = [
        {"id": 1, "text": "Machine learning is fascinating", "category": "tech"},
        {"id": 2, "text": "I love cooking pasta", "category": "food"},
        {"id": 3, "text": "The weather is beautiful today", "category": "general"},
        {"id": 4, "text": "Python is a great language", "category": "tech"},
        {"id": 5, "text": "Pizza is my favorite food", "category": "food"},
    ]
    
    print(f"Created simple dataset with {len(simple_data)} entries")
    print_example_output("First entry", simple_data[0])
    
    # Part 2: Dataset Processing Operator
    print("\n" + "="*50)
    print("🔄 Part 2: Dataset Processing with Operators")
    print("="*50 + "\n")
    
    class DataProcessor(operators.Operator):
        """Processes dataset entries."""
        
        specification = operators.Specification()
        
        def __init__(self, *, add_length: bool = True, lowercase: bool = True):
            self.add_length = add_length
            self.lowercase = lowercase
        
        def forward(self, *, inputs):
            entry = inputs.get("entry", {})
            
            # Process the entry
            processed = entry.copy()
            
            if "text" in processed:
                if self.lowercase:
                    processed["text_lower"] = processed["text"].lower()
                if self.add_length:
                    processed["text_length"] = len(processed["text"])
            
            # Add processing metadata
            processed["processed"] = True
            
            return processed
    
    # Process dataset
    processor = DataProcessor()
    processed_data = []
    
    for entry in simple_data:
        result = processor(entry=entry)
        processed_data.append(result)
    
    print("Processed dataset:")
    print_example_output("Original", simple_data[0])
    print_example_output("Processed", processed_data[0])
    
    # Part 3: Streaming Datasets
    print("\n" + "="*50)
    print("🌊 Part 3: Streaming Large Datasets")
    print("="*50 + "\n")
    
    class StreamingDataset(operators.Operator):
        """Simulates a streaming dataset."""
        
        specification = operators.Specification()
        
        def __init__(self, *, size: int = 1000):
            self.size = size
        
        def generate_entry(self, idx: int) -> Dict[str, Any]:
            """Generate a single data entry."""
            categories = ["tech", "science", "health", "business", "entertainment"]
            topics = {
                "tech": ["AI", "programming", "cloud", "data"],
                "science": ["physics", "biology", "chemistry", "astronomy"],
                "health": ["fitness", "nutrition", "medicine", "wellness"],
                "business": ["startup", "finance", "marketing", "leadership"],
                "entertainment": ["movies", "music", "games", "books"]
            }
            
            category = categories[idx % len(categories)]
            topic = topics[category][idx % len(topics[category])]
            
            return {
                "id": idx,
                "text": f"This is about {topic} in {category}",
                "category": category,
                "topic": topic,
                "score": (idx % 100) / 100.0
            }
        
        def stream(self) -> Iterator[Dict[str, Any]]:
            """Stream data entries."""
            for i in range(self.size):
                yield self.generate_entry(i)
        
        def forward(self, *, inputs):
            batch_size = inputs.get("batch_size", 10)
            offset = inputs.get("offset", 0)
            
            # Return a batch
            batch = []
            for i in range(offset, min(offset + batch_size, self.size)):
                batch.append(self.generate_entry(i))
            
            return {
                "batch": batch,
                "batch_size": len(batch),
                "has_more": offset + batch_size < self.size
            }
    
    # Create streaming dataset
    streamer = StreamingDataset(size=100)
    
    # Process in batches
    print("Processing streaming dataset in batches:")
    total_processed = 0
    offset = 0
    
    while True:
        batch_result = streamer(batch_size=20, offset=offset)
        batch = batch_result["batch"]
        
        if not batch:
            break
        
        total_processed += len(batch)
        print(f"  Processed batch: {len(batch)} entries (total: {total_processed})")
        
        offset += len(batch)
        
        if not batch_result["has_more"]:
            break
    
    # Part 4: Data Filtering and Transformation
    print("\n" + "="*50)
    print("🎯 Part 4: Filtering and Transformation")
    print("="*50 + "\n")
    
    class DataFilter(operators.Operator):
        """Filters dataset based on criteria."""
        
        specification = operators.Specification()
        
        def __init__(self, *, category: str = None, min_score: float = None):
            self.category = category
            self.min_score = min_score
        
        def forward(self, *, inputs):
            data = inputs.get("data", [])
            
            filtered = []
            for entry in data:
                # Apply filters
                if self.category and entry.get("category") != self.category:
                    continue
                if self.min_score and entry.get("score", 0) < self.min_score:
                    continue
                
                filtered.append(entry)
            
            return {
                "filtered_data": filtered,
                "original_count": len(data),
                "filtered_count": len(filtered),
                "filter_rate": len(filtered) / len(data) if data else 0
            }
    
    class DataAggregator(operators.Operator):
        """Aggregates data statistics."""
        
        specification = operators.Specification()
        
        def forward(self, *, inputs):
            data = inputs.get("data", [])
            
            if not data:
                return {"stats": {}, "count": 0}
            
            # Calculate statistics
            categories = {}
            total_score = 0
            
            for entry in data:
                cat = entry.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
                total_score += entry.get("score", 0)
            
            return {
                "stats": {
                    "total_entries": len(data),
                    "categories": categories,
                    "avg_score": total_score / len(data) if data else 0,
                    "unique_categories": len(categories)
                },
                "count": len(data)
            }
    
    # Generate sample data
    sample_batch = streamer(batch_size=50, offset=0)["batch"]
    
    # Filter tech entries
    tech_filter = DataFilter(category="tech")
    filtered = tech_filter(data=sample_batch)
    
    print(f"Tech filtering:")
    print_example_output("Original count", filtered["original_count"])
    print_example_output("Filtered count", filtered["filtered_count"])
    print_example_output("Filter rate", f"{filtered['filter_rate']:.1%}")
    
    # Aggregate statistics
    aggregator = DataAggregator()
    stats = aggregator(data=sample_batch)
    
    print(f"\nDataset statistics:")
    for key, value in stats["stats"].items():
        print_example_output(key, value)
    
    # Part 5: Complete Data Pipeline
    print("\n" + "="*50)
    print("🏗️ Part 5: Complete Data Processing Pipeline")
    print("="*50 + "\n")
    
    class DataPipeline(operators.Operator):
        """Complete data processing pipeline."""
        
        specification = operators.Specification()
        
        def __init__(self):
            self.processor = DataProcessor()
            self.filter = DataFilter(min_score=0.5)
            self.aggregator = DataAggregator()
        
        def forward(self, *, inputs):
            dataset_size = inputs.get("dataset_size", 100)
            batch_size = inputs.get("batch_size", 20)
            
            # Create dataset
            dataset = StreamingDataset(size=dataset_size)
            
            # Process in batches
            all_processed = []
            offset = 0
            
            while offset < dataset_size:
                # Get batch
                batch_result = dataset(batch_size=batch_size, offset=offset)
                batch = batch_result["batch"]
                
                # Process each entry
                processed_batch = []
                for entry in batch:
                    processed = self.processor(entry=entry)
                    processed_batch.append(processed)
                
                # Filter batch
                filtered_result = self.filter(data=processed_batch)
                all_processed.extend(filtered_result["filtered_data"])
                
                offset += len(batch)
            
            # Final aggregation
            final_stats = self.aggregator(data=all_processed)
            
            return {
                "total_processed": len(all_processed),
                "statistics": final_stats["stats"],
                "pipeline_stages": ["load", "process", "filter", "aggregate"]
            }
    
    # Run the pipeline
    pipeline = DataPipeline()
    result = pipeline(dataset_size=200, batch_size=50)
    
    print("Data Pipeline Results:")
    print_example_output("Total processed", result["total_processed"])
    print("\nStatistics:")
    for key, value in result["statistics"].items():
        print_example_output(f"  {key}", value)
    
    # Part 6: Tips for Real Datasets
    print("\n" + "="*50)
    print("💡 Part 6: Working with Real Datasets")
    print("="*50 + "\n")
    
    print("When using Ember's data API with real datasets:")
    print("\n1. Use the dataset registry:")
    print("   dataset = data.load_dataset('mmlu', split='test')")
    
    print("\n2. Stream large datasets:")
    print("   for batch in dataset.stream(batch_size=32):")
    print("       process_batch(batch)")
    
    print("\n3. Transform on the fly:")
    print("   dataset = dataset.map(transform_fn)")
    print("   dataset = dataset.filter(filter_fn)")
    
    print("\n4. Cache processed data:")
    print("   dataset = dataset.cache('processed_data.pkl')")
    
    print("\n" + "="*50)
    print("✅ Key Takeaways")
    print("="*50)
    print("\n1. Process data in batches for memory efficiency")
    print("2. Use operators for reusable data transformations")
    print("3. Stream large datasets instead of loading all at once")
    print("4. Filter early to reduce processing overhead")
    print("5. Aggregate statistics for data understanding")
    print("6. Build pipelines for complex workflows")
    
    print("\nNext: Explore streaming_data.py for advanced patterns!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())