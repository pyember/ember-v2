"""Data loading and streaming for machine learning datasets.

This module provides a unified interface for loading and processing datasets
from various sources including HuggingFace, local files, and custom sources.
The API emphasizes streaming by default for memory efficiency while providing
explicit eager loading when needed.

The design follows progressive disclosure: simple usage for common cases,
with advanced functionality available through method chaining when needed.

Typical usage:
    Basic streaming::

        for item in stream("mmlu"):
            print(item["question"], item["answer"])

    Loading with filters::

        physics_items = load("mmlu",
                            subset="physics",
                            filter=lambda x: x["metadata"]["difficulty"] == "hard")

    Advanced chaining::

        results = (stream("squad")
                  .filter(lambda x: len(x["question"]) > 50)
                  .transform(lambda x: {**x, "prompt": format_prompt(x)})
                  .first(100))

    Custom data sources::

        class APIDataSource:
            def read_batches(self, batch_size=32):
                for page in fetch_pages():
                    yield page["items"]

        register("api_data", APIDataSource())
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)


@runtime_checkable
class DataSource(Protocol):
    """Protocol defining the interface for data sources.

    Any class implementing this protocol can be used as a data source
    for the streaming and loading functions. The protocol requires only
    one method: read_batches.

    Example implementation::

        class CustomSource:
            def __init__(self, data: List[Dict]):
                self.data = data

            def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
                for i in range(0, len(self.data), batch_size):
                    yield self.data[i:i + batch_size]
    """

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches of dictionaries from the data source.

        Args:
            batch_size: Number of items per batch. Implementations should
                respect this for memory efficiency.

        Yields:
            Lists of dictionaries, each list containing up to batch_size items.
            The final batch may contain fewer items.
        """
        ...


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about a dataset.

    Contains essential information for understanding dataset characteristics
    and making decisions about processing strategies.

    Attributes:
        name: Unique identifier for the dataset.
        description: Human-readable description of the dataset's purpose.
        size_bytes: Total size in bytes (0 if unknown).
        example_count: Number of examples (0 if unknown).
        example_item: A single example item showing the data schema.
        streaming_supported: Whether the dataset supports streaming.
    """

    name: str
    description: str
    size_bytes: int
    example_count: int
    example_item: Dict[str, Any]
    streaming_supported: bool = True


def stream(
    source: Union[str, DataSource],
    *,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    batch_size: int = 32,
    max_items: Optional[int] = None,
    normalize: bool = True,
) -> StreamIterator:
    """Stream data from a source with optional processing.

    Provides memory-efficient iteration over datasets with support for
    filtering, transformation, and limiting. Returns a StreamIterator
    that supports method chaining for complex pipelines.

    Args:
        source: Dataset name (e.g., "mmlu") or DataSource instance.
        subset: Dataset configuration or subset name (e.g., "high_school_physics").
        split: Dataset split to load ("train", "validation", or "test").
        filter: Predicate function to filter items. Items are kept when
            the function returns True.
        transform: Function to transform each item. Receives a dict and
            should return a dict.
        batch_size: Number of items to load per batch for efficiency.
            Does not affect iteration interface.
        max_items: Maximum number of items to yield. None for unlimited.
        normalize: Whether to normalize items to standard schema with
            "question", "answer", "choices", and "metadata" fields.

    Returns:
        StreamIterator that yields dictionaries and supports chaining.

    Raises:
        ValueError: If the dataset name is not found in registry.
        FileNotFoundError: If a FileSource points to missing file.
        ImportError: If HuggingFace datasets not installed for HF sources.

    Examples:
        Basic iteration over a dataset::

            for item in stream("mmlu"):
                print(f"Q: {item['question']}")
                print(f"A: {item['answer']}")

        Loading a specific subset and split::

            for item in stream("mmlu", subset="anatomy", split="test"):
                evaluate(item)

        Filtering items inline::

            for item in stream("squad",
                             filter=lambda x: len(x["question"]) > 100):
                process_long_question(item)

        Transforming items inline::

            for item in stream("gsm8k",
                             transform=lambda x: {
                                 **x,
                                 "prompt": f"Solve: {x['question']}"
                             }):
                print(item["prompt"])

        Using method chaining for complex pipelines::

            hard_physics = (stream("mmlu", subset="physics")
                           .filter(lambda x: x["metadata"]["difficulty"] > 3)
                           .transform(add_few_shot_examples)
                           .limit(50))

            for item in hard_physics:
                evaluate_with_cot(item)

        Custom data source::

            class MyData:
                def read_batches(self, batch_size=32):
                    yield [{"question": "Q1", "answer": "A1"}]

            for item in stream(MyData()):
                print(item)
    """
    # Resolve source to DataSource instance
    if isinstance(source, str):
        data_source = _registry.get_source(source, subset, split)
    else:
        data_source = source

    # Create iterator with all configuration
    return StreamIterator(
        source=data_source,
        filter=filter,
        transform=transform,
        batch_size=batch_size,
        max_items=max_items,
        normalize=normalize,
    )


class StreamIterator:
    """Iterator over data with support for chaining operations.

    Created by the stream() function, this class provides iteration over
    data items with support for progressive enhancement through method
    chaining. All methods return new iterators, allowing immutable
    operation chains.

    The iterator is lazy - no data is loaded until iteration begins,
    and transformations are applied on-the-fly for memory efficiency.

    Examples:
        Basic iteration::

            for item in stream("mmlu"):
                print(item["question"])

        Chaining operations::

            filtered = stream("mmlu").filter(lambda x: "physics" in x["question"])
            transformed = filtered.transform(lambda x: {**x, "difficulty": 5})
            first_ten = transformed.limit(10)

            for item in first_ten:
                print(item)

        Collecting results::

            # Get first 5 items as a list
            items = stream("mmlu").first(5)

            # Collect all filtered items (caution with large datasets)
            physics = stream("mmlu").filter(is_physics).collect()
    """

    def __init__(
        self,
        source: DataSource,
        *,
        filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        batch_size: int = 32,
        max_items: Optional[int] = None,
        normalize: bool = True,
    ):
        """Initialize iterator with source and processing configuration.

        Args:
            source: DataSource to read from.
            filter: Optional filter predicate.
            transform: Optional transformation function.
            batch_size: Batch size for reading from source.
            max_items: Maximum items to yield.
            normalize: Whether to normalize items.
        """
        self._source = source
        self._filter = filter
        self._transform = transform
        self._batch_size = batch_size
        self._max_items = max_items
        self._normalize = normalize

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over items with all processing applied.

        Yields:
            Dictionaries representing data items, with normalization,
            filtering, and transformation applied as configured.
        """
        count = 0

        for batch in self._source.read_batches(self._batch_size):
            for item in batch:
                # Apply max_items limit
                if self._max_items is not None and count >= self._max_items:
                    return

                # Normalize if requested
                if self._normalize:
                    item = _normalize(item)

                # Apply filter
                if self._filter and not self._filter(item):
                    continue

                # Apply transformation
                if self._transform:
                    item = self._transform(item)

                yield item
                count += 1

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> StreamIterator:
        """Create a new iterator with an additional filter.

        Multiple filters are combined with AND logic - an item must pass
        all filters to be yielded.

        Args:
            predicate: Function that returns True to keep an item.

        Returns:
            New StreamIterator with the filter applied.

        Examples:
            Single filter::

                physics = stream("mmlu").filter(lambda x: x["metadata"]["subject"] == "physics")

            Multiple filters::

                hard_physics = (stream("mmlu")
                               .filter(lambda x: x["metadata"]["subject"] == "physics")
                               .filter(lambda x: x["metadata"]["difficulty"] > 3))

            Complex filter::

                def has_equation(item):
                    return any(c in item["question"] for c in "=∫∑∏")

                math_questions = stream("mmlu").filter(has_equation)
        """
        # Combine with existing filter using AND logic
        if self._filter:
            old_filter = self._filter
            def new_filter(x):
                return old_filter(x) and predicate(x)
        else:
            new_filter = predicate

        return StreamIterator(
            source=self._source,
            filter=new_filter,
            transform=self._transform,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize=self._normalize,
        )

    def transform(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> StreamIterator:
        """Create a new iterator with an additional transformation.

        Multiple transformations are applied in order - each transformation
        receives the output of the previous one.

        Args:
            fn: Function that transforms an item dictionary.

        Returns:
            New StreamIterator with the transformation applied.

        Examples:
            Add a field::

                with_id = stream("mmlu").transform(
                    lambda x: {**x, "id": generate_uuid()}
                )

            Format for prompting::

                prompted = stream("gsm8k").transform(
                    lambda x: {
                        **x,
                        "prompt": f"Problem: {x['question']}\\nSolution:"
                    }
                )

            Chain transformations::

                processed = (stream("squad")
                            .transform(clean_whitespace)
                            .transform(add_context_length)
                            .transform(format_for_model))
        """
        # Combine with existing transformation
        if self._transform:
            old_transform = self._transform
            def new_transform(x):
                return fn(old_transform(x))
        else:
            new_transform = fn

        return StreamIterator(
            source=self._source,
            filter=self._filter,
            transform=new_transform,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize=self._normalize,
        )

    def limit(self, n: int) -> StreamIterator:
        """Create a new iterator that yields at most n items.

        If the iterator already has a limit, the minimum is used.

        Args:
            n: Maximum number of items to yield.

        Returns:
            New StreamIterator with the limit applied.

        Examples:
            Get first 10 items::

                for item in stream("mmlu").limit(10):
                    print(item["question"])

            Combine with filtering::

                # Get up to 5 physics questions
                physics_sample = (stream("mmlu")
                                 .filter(lambda x: "physics" in x["metadata"]["subject"])
                                 .limit(5))
        """
        current_max = self._max_items
        if current_max is None:
            new_max = n
        else:
            new_max = min(current_max, n)

        return StreamIterator(
            source=self._source,
            filter=self._filter,
            transform=self._transform,
            batch_size=self._batch_size,
            max_items=new_max,
            normalize=self._normalize,
        )

    def first(self, n: int) -> List[Dict[str, Any]]:
        """Get the first n items as a list.

        Convenience method equivalent to list(iterator.limit(n)).

        Args:
            n: Number of items to get.

        Returns:
            List containing up to n items.

        Examples:
            Get first 5 items::

                items = stream("mmlu").first(5)
                print(f"Got {len(items)} items")

            Get filtered sample::

                hard_questions = (stream("mmlu")
                                 .filter(lambda x: x["metadata"]["difficulty"] > 4)
                                 .first(10))
        """
        return list(self.limit(n))

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all items into a list.

        Warning:
            This loads all data into memory. Use with caution on large
            datasets or infinite streams. Consider using limit() first
            or processing items one at a time.

        Returns:
            List of all items from the iterator.

        Examples:
            Collect filtered subset::

                physics_questions = (stream("mmlu")
                                    .filter(lambda x: x["metadata"]["subject"] == "physics")
                                    .collect())

            Safe collection with limit::

                sample = stream("large_dataset").limit(1000).collect()
        """
        return list(self)


def load(
    source: Union[str, DataSource],
    *,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    max_items: Optional[int] = None,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """Load data into memory as a list.

    Convenience function that streams data and collects it into a list.
    Equivalent to stream(...).collect() but makes the memory usage explicit
    in the function name.

    Args:
        source: Dataset name or DataSource instance.
        subset: Dataset configuration or subset name.
        split: Dataset split ("train", "validation", or "test").
        filter: Predicate function to filter items.
        transform: Function to transform each item.
        max_items: Maximum number of items to load.
        normalize: Whether to normalize items to standard schema.

    Returns:
        List of dictionaries containing the loaded data.

    Raises:
        ValueError: If dataset not found.
        MemoryError: If dataset too large for memory.

    Examples:
        Load a small dataset::

            data = load("squad", split="validation")
            print(f"Loaded {len(data)} examples")

        Load with filtering::

            physics = load("mmlu",
                          subset="high_school_physics",
                          filter=lambda x: x["metadata"]["difficulty"] > 2)

        Load limited sample::

            sample = load("large_dataset", max_items=1000)

        Load and transform::

            prompted = load("gsm8k",
                           transform=lambda x: {
                               **x,
                               "input": format_prompt(x["question"])
                           })
    """
    return list(
        stream(
            source,
            subset=subset,
            split=split,
            filter=filter,
            transform=transform,
            max_items=max_items,
            normalize=normalize,
        )
    )


def metadata(dataset: str) -> DatasetInfo:
    """Get metadata for a registered dataset.

    Args:
        dataset: Name of the dataset to get metadata for.

    Returns:
        DatasetInfo object with dataset metadata.

    Raises:
        ValueError: If dataset not found in registry.

    Examples:
        Check dataset size::

            info = metadata("mmlu")
            size_gb = info.size_bytes / 1e9 if info.size_bytes else "Unknown"
            print(f"Dataset size: {size_gb} GB")
            print(f"Example count: {info.example_count:,}")

        Examine data schema::

            info = metadata("squad")
            print("Example item:")
            print(json.dumps(info.example_item, indent=2))

        Check streaming support::

            info = metadata("custom_dataset")
            if info.streaming_supported:
                print("Streaming supported - memory efficient")
            else:
                print("No streaming - will load all data")
    """
    return _registry.get_metadata(dataset)


def list_datasets() -> List[str]:
    """List all registered dataset names.

    Returns:
        Sorted list of dataset names available for loading.

    Examples:
        Show available datasets::

            datasets = list_datasets()
            print(f"Available datasets: {len(datasets)}")
            for name in datasets:
                print(f"  - {name}")

        Check if dataset exists::

            if "mmlu" in list_datasets():
                data = stream("mmlu")
    """
    return _registry.list_available()


def register(name: str, source: DataSource, metadata: Optional[DatasetInfo] = None) -> None:
    """Register a custom data source.

    Makes a data source available for loading by name through the
    stream() and load() functions.

    Args:
        name: Unique name to register the source under.
        source: DataSource implementation.
        metadata: Optional metadata about the dataset.

    Raises:
        TypeError: If source doesn't implement DataSource protocol.

    Examples:
        Register a file source::

            register("my_data", FileSource("data/train.jsonl"))

            # Now usable by name
            for item in stream("my_data"):
                process(item)

        Register a custom source::

            class RedisSource:
                def __init__(self, key_pattern):
                    self.pattern = key_pattern
                    self.redis = redis.Redis()

                def read_batches(self, batch_size=32):
                    keys = self.redis.keys(self.pattern)
                    for i in range(0, len(keys), batch_size):
                        batch = []
                        for key in keys[i:i+batch_size]:
                            batch.append(json.loads(self.redis.get(key)))
                        yield batch

            register("redis_data", RedisSource("items:*"))

        Register with metadata::

            register(
                "custom_qa",
                FileSource("qa_data.json"),
                DatasetInfo(
                    name="custom_qa",
                    description="Internal QA dataset",
                    size_bytes=1024000,
                    example_count=5000,
                    example_item={"question": "...", "answer": "..."},
                    streaming_supported=True
                )
            )
    """
    if not isinstance(source, DataSource):
        raise TypeError(
            f"Source must implement DataSource protocol. " f"Got {type(source).__name__}"
        )
    _registry.register(name, source, metadata)


def from_file(path: Union[str, Path], **kwargs) -> StreamIterator:
    """Stream data from a file.

    Convenience function for streaming from files without explicit
    FileSource creation. Supports JSON, JSONL, and CSV formats.

    Args:
        path: Path to file. Format detected from extension.
        **kwargs: Additional arguments passed to stream().

    Returns:
        StreamIterator over file contents.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format not supported.

    Examples:
        Stream from JSONL::

            for item in from_file("data.jsonl"):
                print(item["text"])

        Stream from CSV with filtering::

            for item in from_file("data.csv",
                                filter=lambda x: float(x["score"]) > 0.8):
                process(item)

        Stream with transformation::

            for item in from_file("raw_data.json",
                                transform=lambda x: {
                                    "input": x["text"],
                                    "label": x["category"]
                                }):
                train_model(item)
    """
    return stream(FileSource(path), **kwargs)


def load_file(path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
    """Load file data into memory.

    Convenience function for loading files without explicit FileSource
    creation. Supports JSON, JSONL, and CSV formats.

    Args:
        path: Path to file. Format detected from extension.
        **kwargs: Additional arguments passed to load().

    Returns:
        List of dictionaries from file.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format not supported.
        MemoryError: If file too large.

    Examples:
        Load JSON file::

            data = load_file("config.json")
            print(f"Loaded {len(data)} items")

        Load CSV with limit::

            sample = load_file("large_data.csv", max_items=100)

        Load and filter::

            valid_items = load_file("data.jsonl",
                                  filter=lambda x: x.get("valid", False))
    """
    return load(FileSource(path), **kwargs)


def _normalize(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize item to standard schema.

    Converts various dataset formats to a consistent schema with
    standard field names. Handles common variations in field naming
    across different datasets.

    Standard schema:
        - question: The question, query, or prompt text
        - answer: The answer, target, or label
        - choices: Multiple choice options as a dictionary
        - metadata: All other fields

    Args:
        item: Dictionary to normalize.

    Returns:
        Normalized dictionary with standard fields.

    Examples:
        >>> _normalize({"query": "What is 2+2?", "target": "4"})
        {"question": "What is 2+2?", "answer": "4", "choices": {}, "metadata": {}}

        >>> _normalize({"question": "Pick one", "options": ["A", "B", "C"]})
        {"question": "Pick one", "answer": "", "choices": {"A": "A", "B": "B", "C": "C"},
         "metadata": {}}
    """
    # Extract question with fallbacks
    question = (
        item.get("question")
        or item.get("query")
        or item.get("prompt")
        or item.get("text")
        or item.get("input", "")
    )

    # Extract answer with fallbacks
    answer = (
        item.get("answer")
        or item.get("target")
        or item.get("label")
        or item.get("output")
        or item.get("response", "")
    )

    # Extract choices, converting list to dict if needed
    choices = item.get("choices") or item.get("options", {})
    if isinstance(choices, list):
        # Convert ["opt1", "opt2"] to {"A": "opt1", "B": "opt2"}
        choices = {chr(65 + i): choice for i, choice in enumerate(choices)}
    elif not isinstance(choices, dict):
        choices = {}

    # Build normalized item
    normalized = {
        "question": question,
        "answer": answer,
        "choices": choices,
        "metadata": {},
    }

    # Handle existing metadata field
    if "metadata" in item and isinstance(item["metadata"], dict):
        normalized["metadata"].update(item["metadata"])

    # Add remaining fields to metadata
    excluded_keys = {
        "question",
        "query",
        "prompt",
        "text",
        "input",
        "answer",
        "target",
        "label",
        "output",
        "response",
        "choices",
        "options",
        "metadata",
    }

    for key, value in item.items():
        if key not in excluded_keys:
            normalized["metadata"][key] = value

    return normalized


class HuggingFaceSource:
    """Data source for HuggingFace datasets.

    Provides streaming access to datasets from the HuggingFace Hub.
    Requires the 'datasets' package to be installed.

    Examples:
        Load a standard dataset::

            source = HuggingFaceSource("squad", split="validation")
            for batch in source.read_batches():
                process_batch(batch)

        Load with configuration::

            source = HuggingFaceSource("super_glue", config="boolq", split="train")

        Use with register::

            register("my_hf_dataset",
                    HuggingFaceSource("username/dataset-name", split="test"))
    """

    def __init__(self, name: str, split: Optional[str] = None, config: Optional[str] = None):
        """Initialize HuggingFace source.

        Args:
            name: Dataset name on HuggingFace Hub.
            split: Dataset split. Defaults to "train".
            config: Dataset configuration name for multi-config datasets.
        """
        self.name = name
        self.split = split or "train"
        self.config = config
        self._dataset = None

    def with_config(
        self, subset: Optional[str] = None, split: Optional[str] = None
    ) -> HuggingFaceSource:
        """Create a new source with different configuration.

        Args:
            subset: New configuration name.
            split: New split name.

        Returns:
            New HuggingFaceSource with updated configuration.
        """
        return HuggingFaceSource(
            name=self.name, split=split or self.split, config=subset or self.config
        )

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Read batches from HuggingFace dataset.

        Args:
            batch_size: Number of items per batch.

        Yields:
            Lists of dictionaries from the dataset.

        Raises:
            ImportError: If datasets package not installed.
            ValueError: If dataset not found.
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets not installed. Install with: pip install datasets"
            ) from e

        # Load dataset lazily with streaming
        if self._dataset is None:
            self._dataset = load_dataset(self.name, self.config, split=self.split, streaming=True)

        # Yield batches
        batch = []
        for item in self._dataset:
            batch.append(dict(item))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield final partial batch
        if batch:
            yield batch


class FileSource:
    """Data source for local files.

    Supports JSON, JSONL, and CSV file formats. Format is detected
    from file extension.

    File format details:
        - JSON: Expects array of objects or single object
        - JSONL: One JSON object per line
        - CSV: First row used as headers

    Examples:
        Load JSONL file::

            source = FileSource("data/train.jsonl")
            for batch in source.read_batches(batch_size=64):
                train_on_batch(batch)

        Load CSV file::

            source = FileSource("data/results.csv")
            register("results", source)

        Use with Path object::

            from pathlib import Path
            source = FileSource(Path.home() / "data" / "test.json")
    """

    def __init__(self, path: Union[Path, str]):
        """Initialize file source.

        Args:
            path: Path to file. Must exist and have supported extension.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Read batches from file.

        Args:
            batch_size: Number of items per batch.

        Yields:
            Lists of dictionaries from file.

        Raises:
            ValueError: If file format not supported.
            json.JSONDecodeError: If JSON/JSONL malformed.
        """
        suffix = self.path.suffix.lower()

        if suffix == ".jsonl":
            yield from self._read_jsonl(batch_size)
        elif suffix == ".json":
            yield from self._read_json(batch_size)
        elif suffix == ".csv":
            yield from self._read_csv(batch_size)
        else:
            raise ValueError(f"Unsupported file type: {suffix}\n" f"Supported: .json, .jsonl, .csv")

    def _read_jsonl(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Read JSONL file in batches."""
        batch = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}", e.doc, e.pos
                    ) from e

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def _read_json(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Read JSON file in batches."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Yield batches from list
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]
        elif isinstance(data, dict):
            # Single object
            yield [data]
        else:
            raise ValueError(f"JSON file must contain array or object, got {type(data).__name__}")

    def _read_csv(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Read CSV file in batches."""
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            batch = []

            for row in reader:
                # Convert to regular dict to ensure JSON serializable
                batch.append(dict(row))

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch


class _RegistryClass:
    """Internal registry for named datasets.

    Thread-safe storage for dataset sources and metadata with
    automatic initialization of common datasets.
    """

    def __init__(self):
        """Initialize empty registry with thread lock."""
        self._sources: Dict[str, DataSource] = {}
        self._metadata: Dict[str, DatasetInfo] = {}
        self._lock = threading.Lock()
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Register commonly used datasets."""
        # Standard evaluation datasets
        standard_datasets = {
            "mmlu": ("cais/mmlu", "test", "all"),
            "squad": ("squad", "validation", None),
            "gsm8k": ("gsm8k", "test", "main"),
            "hellaswag": ("Rowan/hellaswag", "validation", None),
            "truthfulqa": ("truthful_qa", "validation", "multiple_choice"),
            "arc": ("ai2_arc", "test", "ARC-Challenge"),
            "winogrande": ("winogrande", "validation", "winogrande_xl"),
        }

        for name, (hf_name, split, config) in standard_datasets.items():
            source = HuggingFaceSource(hf_name, split, config)
            self.register(name, source)

    def register(
        self, name: str, source: DataSource, metadata: Optional[DatasetInfo] = None
    ) -> None:
        """Register a data source with optional metadata."""
        with self._lock:
            self._sources[name] = source
            if metadata:
                self._metadata[name] = metadata

    def get_source(
        self, name: str, subset: Optional[str] = None, split: Optional[str] = None
    ) -> DataSource:
        """Get source, trying registry first then HuggingFace."""
        with self._lock:
            if name in self._sources:
                source = self._sources[name]
                # Handle sources that support configuration
                if hasattr(source, "with_config"):
                    return source.with_config(subset=subset, split=split)
                return source

        # Try as HuggingFace dataset if not in registry
        return HuggingFaceSource(name, split=split, config=subset)

    def get_metadata(self, name: str) -> DatasetInfo:
        """Get or generate metadata for a dataset."""
        with self._lock:
            if name in self._metadata:
                return self._metadata[name]

        # Generate metadata by loading one example
        try:
            example = None
            for item in stream(name, max_items=1, normalize=False):
                example = item
                break

            metadata = DatasetInfo(
                name=name,
                description=f"Dataset: {name}",
                size_bytes=0,  # Unknown without full scan
                example_count=0,  # Unknown without full scan
                example_item=example or {},
                streaming_supported=True,
            )

            with self._lock:
                self._metadata[name] = metadata

            return metadata

        except Exception:
            # Return minimal metadata on error
            return DatasetInfo(
                name=name,
                description=f"Dataset: {name} (error loading)",
                size_bytes=0,
                example_count=0,
                example_item={},
                streaming_supported=True,
            )

    def list_available(self) -> List[str]:
        """List all registered dataset names."""
        with self._lock:
            return sorted(list(self._sources.keys()))


# Global registry instance
_registry = _RegistryClass()


__all__ = [
    # Main functions
    "stream",
    "load",
    "metadata",
    "list_datasets",
    "register",
    # File convenience functions
    "from_file",
    "load_file",
    # Types and classes
    "DataSource",
    "DatasetInfo",
    "StreamIterator",
    "FileSource",
    "HuggingFaceSource",
]
