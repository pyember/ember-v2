"""Data API for Ember.

This module provides a facade for Ember's data processing system with a streamlined
interface for loading, transforming, and working with datasets.

Examples:
    # Discovering available datasets
    from ember.api import data
    available_datasets = data.list()  # Returns ["mmlu", "codeforces", "gpqa", ...]
    
    # Getting detailed dataset information
    mmlu_info = data.info("mmlu")
    print(f"Description: {mmlu_info.description}")
    print(f"Task Type: {mmlu_info.task_type}")
    print(f"Splits: {mmlu_info.splits}")  # ["train", "test", "validation"]
    print(f"Subjects: {mmlu_info.subjects}")  # ["physics", "mathematics", ...]
    
    # Exploring dataset structure (first few items to understand schema)
    sample = list(data("mmlu", streaming=True, limit=3))
    first_item = sample[0]
    print(f"Available fields: {dir(first_item)}")  # See all accessible attributes
    
    # Loading a dataset directly
    from ember.api import data
    mmlu_data = data("mmlu")

    # Using the builder pattern with transformations
    from ember.api import data
    dataset = (
        data.builder()
        .from_registry("mmlu")
        .subset("physics")
        .split("test")
        .sample(100)
        .transform(lambda x: {"query": f"Question: {x['question']}"})
        .build()
    )

    # Memory-efficient streaming
    for item in data("mmlu", streaming=True):
        process(item)

    # Accessing dataset entries
    for entry in dataset:
        print(f"Question: {entry.question}")

    # Registering a custom dataset
    from ember.api import data, TaskType

    data.register(
        name="custom_qa",
        source="custom/qa",
        task_type=TaskType.QUESTION_ANSWERING
    )
"""

import threading
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

from ember.core.context.ember_context import EmberContext
from ember.core.utils.data.base.config import BaseDatasetConfig as DatasetConfig
from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from ember.core.utils.data.base.transformers import IDatasetTransformer
from ember.core.utils.data.context.data_context import DataContext
from ember.core.utils.data.registry import register

# Registry will be initialized when DataContext is created

# Type variables for generic typing
T = TypeVar("T")
U = TypeVar("U")


class DataItem:
    """Normalized representation of a dataset item.

    This class provides consistent attribute access across different dataset formats,
    making it simple to work with items regardless of their underlying representation.

    Attributes:
        question: The question text with consistent naming
        options: Answer options with consistent naming
        answer: The correct answer with consistent naming
    """

    def __init__(self, entry: Any) -> None:
        """Initialize with a dataset entry.

        Args:
            entry: The dataset entry to wrap (DatasetEntry, dict, or compatible object)
        """
        self._entry = entry
        self._normalized = self._normalize(entry)

    def _normalize(self, entry: Any) -> Dict[str, Any]:
        """Convert entry to a normalized dictionary.

        Args:
            entry: The dataset entry to normalize

        Returns:
            Normalized dictionary representation
        """
        if isinstance(entry, dict):
            return entry

        normalized = {}

        # Handle DatasetEntry from Ember (prioritize this)
        if isinstance(entry, DatasetEntry):
            # Copy content dictionary
            if hasattr(entry, "content") and isinstance(entry.content, dict):
                normalized.update(entry.content)

            # Copy metadata dictionary
            if hasattr(entry, "metadata") and isinstance(entry.metadata, dict):
                normalized.update(entry.metadata)

            # Copy direct attributes with consistent naming
            if hasattr(entry, "query"):
                normalized["question"] = entry.query
            if hasattr(entry, "choices"):
                normalized["options"] = entry.choices

            return normalized

        # Handle other object types
        if hasattr(entry, "__dict__"):
            normalized.update(entry.__dict__)

        # Extract important attributes if present
        for src, dst in [
            ("query", "question"),
            ("question", "question"),
            ("choices", "options"),
            ("options", "options"),
            ("answer", "answer"),
            ("correct_answer", "answer"),
        ]:
            if hasattr(entry, src):
                normalized[dst] = getattr(entry, src)

        return normalized

    @property
    def question(self) -> Optional[str]:
        """Get question text with consistent naming.

        Returns:
            The question text or None if not found
        """
        return self._normalized.get("question")

    @property
    def options(self) -> Dict[str, str]:
        """Get answer options with consistent naming.

        Returns:
            Dictionary of answer options or empty dict if not found
        """
        options = self._normalized.get("options", {})
        return options if isinstance(options, dict) else {}

    @property
    def answer(self) -> Optional[str]:
        """Get correct answer with consistent naming.

        Returns:
            The correct answer or None if not found
        """
        return self._normalized.get("answer")

    def __getattr__(self, name: str) -> Any:
        """Access attributes not covered by properties.

        Args:
            name: Attribute name

        Returns:
            Attribute value

        Raises:
            AttributeError: If attribute not found
        """
        # Check normalized dictionary
        if name in self._normalized:
            return self._normalized[name]

        # Check original entry
        if hasattr(self._entry, name):
            return getattr(self._entry, name)

        # Not found
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        """Create string representation.

        Returns:
            String representation of the item
        """
        question = self.question
        if question:
            preview = question[:50] + ("..." if len(question) > 50 else "")
            return f"DataItem('{preview}')"
        return f"DataItem({self._normalized})"


class Dataset(Generic[T]):
    """A dataset representation with convenient access to entries.

    Attributes:
        entries: List of dataset entries
        info: Dataset metadata
    """

    def __init__(self, entries: List[T], info: Optional[DatasetInfo] = None):
        """Initialize a Dataset with entries and optional info.

        Args:
            entries: List of dataset entries
            info: Optional dataset metadata
        """
        self.entries = entries
        self.info = info

    def __getitem__(self, idx: int) -> T:
        """Get a dataset entry by index.

        Args:
            idx: Index of the entry to retrieve

        Returns:
            The dataset entry at the specified index

        Raises:
            IndexError: If the index is out of range
        """
        return self.entries[idx]

    def __iter__(self) -> Iterator[T]:
        """Iterate over dataset entries.

        Returns:
            Iterator over dataset entries
        """
        return iter(self.entries)

    def __len__(self) -> int:
        """Get the number of entries in the dataset.

        Returns:
            Number of entries
        """
        return len(self.entries)


class DatasetBuilder:
    """Builder for dataset loading configuration.

    Provides a fluent interface for specifying dataset parameters and transformations
    before loading. Enables method chaining for concise, readable dataset preparation.
    """

    def __init__(self, *, context: Union[EmberContext, DataContext]) -> None:
        """Initialize dataset builder with default configuration.

        Args:
            context: EmberContext or DataContext for dataset operations
        """
        if isinstance(context, EmberContext):
            # Store EmberContext and get DataContext from it
            self._ember_context = context
            self._data_context = DataContext.create_from_ember_context(context)
        else:
            # Use provided DataContext
            self._data_context = context
            self._ember_context = None

        self._dataset_name: Optional[str] = None
        self._split: Optional[str] = None
        self._sample_size: Optional[int] = None
        self._seed: Optional[int] = None
        self._config: Dict[str, Any] = {}
        self._transformers: List[IDatasetTransformer] = []
        self._streaming: bool = True  # Default to streaming for efficiency
        self._batch_size: Optional[int] = None

    def from_registry(self, dataset_name: str) -> "DatasetBuilder":
        """Specify dataset to load from registry.

        Args:
            dataset_name: Name of the registered dataset

        Returns:
            Self for method chaining

        Raises:
            ValueError: If dataset is not found in registry
        """
        registry = self._data_context.registry
        if not registry.get(name=dataset_name):
            available = registry.list_datasets()
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available datasets: {available}"
            )
        self._dataset_name = dataset_name
        return self

    def subset(self, subset_name: str) -> "DatasetBuilder":
        """Select dataset subset.

        Args:
            subset_name: Name of the subset to select

        Returns:
            Self for method chaining
        """
        self._config["subset"] = subset_name
        return self

    def split(self, split_name: str) -> "DatasetBuilder":
        """Set dataset split.

        Args:
            split_name: Name of the split (e.g., "train", "test", "validation")

        Returns:
            Self for method chaining
        """
        self._split = split_name
        return self

    def sample(self, count: int) -> "DatasetBuilder":
        """Set number of samples to load.

        Args:
            count: Number of samples

        Returns:
            Self for method chaining

        Raises:
            ValueError: If count is negative
        """
        if count < 0:
            raise ValueError(f"Sample count must be non-negative, got {count}")
        self._sample_size = count
        return self

    def seed(self, seed_value: int) -> "DatasetBuilder":
        """Set random seed for reproducible sampling.

        Args:
            seed_value: Random seed value

        Returns:
            Self for method chaining
        """
        self._seed = seed_value
        return self

    def transform(
        self,
        transform_fn: Union[
            Callable[[Dict[str, Any]], Dict[str, Any]], IDatasetTransformer
        ],
    ) -> "DatasetBuilder":
        """Add transformation function to dataset processing pipeline.

        Transformations are applied in the order they're added.

        Args:
            transform_fn: Function that transforms dataset items or transformer instance

        Returns:
            Self for method chaining
        """
        # Import here to avoid circular imports
        from ember.core.utils.data.base.transformers import (
            DatasetType,
            IDatasetTransformer,
        )

        # Use transformer directly if it implements the interface
        if isinstance(transform_fn, IDatasetTransformer):
            self._transformers.append(transform_fn)
            return self

        # Create adapter for function-based transformers
        class FunctionTransformer(IDatasetTransformer):
            """Adapter converting functions to IDatasetTransformer."""

            def __init__(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
                """Initialize with transformation function.

                Args:
                    fn: Function that transforms dataset items
                """
                self._transform_fn = fn

            def transform(self, *, data: DatasetType) -> DatasetType:
                """Apply transformation to dataset.

                Args:
                    data: Dataset to transform

                Returns:
                    Transformed dataset
                """
                if hasattr(data, "map") and callable(getattr(data, "map", None)):
                    # For HuggingFace datasets
                    return data.map(self._transform_fn)  # type: ignore
                if isinstance(data, list):
                    # For list of dictionaries
                    return [self._transform_fn(item) for item in data]
                # For other data structures
                return self._transform_fn(data)  # type: ignore

            def transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
                """Transform a single item.

                Args:
                    item: Item to transform

                Returns:
                    Transformed item
                """
                return self._transform_fn(item)

        self._transformers.append(FunctionTransformer(transform_fn))
        return self

    def config(self, **kwargs) -> "DatasetBuilder":
        """Set additional configuration parameters.

        Args:
            **kwargs: Configuration parameters as keyword arguments

        Returns:
            Self for method chaining
        """
        self._config.update(kwargs)
        return self

    def batch_size(self, size: int) -> "DatasetBuilder":
        """Set batch size for streaming datasets.

        Args:
            size: Batch size (must be positive)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If size is not positive
        """
        if size <= 0:
            raise ValueError(f"Batch size must be positive, got {size}")
        self._batch_size = size
        return self

    def streaming(self, enabled: bool = True) -> "DatasetBuilder":
        """Enable or disable streaming mode.

        Args:
            enabled: Whether to use streaming mode

        Returns:
            Self for method chaining
        """
        self._streaming = enabled
        return self

    def limit(self, count: int) -> "DatasetBuilder":
        """Limit the number of items (alias for sample).

        Args:
            count: Maximum number of items

        Returns:
            Self for method chaining
        """
        return self.sample(count)

    def build(
        self, dataset_name: Optional[str] = None
    ) -> Union[Dataset[DatasetEntry], Iterator[DataItem]]:
        """Build and load dataset with configured parameters.

        Args:
            dataset_name: Name of dataset to load (optional if set via from_registry)

        Returns:
            Loaded dataset with applied transformations or streaming iterator

        Raises:
            ValueError: If dataset name is not provided or dataset not found
        """
        # Determine final dataset name
        final_name = dataset_name or self._dataset_name
        if not final_name:
            raise ValueError(
                "Dataset name must be provided either via build() or from_registry()"
            )

        # Get dataset registry from data context
        registry = self._data_context.registry

        # Get dataset entry from registry
        dataset_entry = registry.get(name=final_name)
        if not dataset_entry:
            available = registry.list_datasets()
            raise ValueError(
                f"Dataset '{final_name}' not found. Available datasets: {available}"
            )

        if self._streaming:
            # Use optimized streaming dataset creation
            dataset = self._data_context.get_streaming_dataset(
                name=final_name,
                transformers=self._transformers,
                batch_size=self._batch_size
                or 32,  # Use builder's batch size or default
            )

            # Apply limit if specified
            if self._sample_size is not None:
                dataset = dataset.limit(self._sample_size)

            # Wrap in DataItem for normalized access
            return map(DataItem, dataset)
        else:
            # Handle Hugging Face dataset-specific configs
            config_name = self._config.get("subset")

            # Create configuration object
            config = DatasetConfig(
                split=self._split,
                sample_size=self._sample_size,
                random_seed=self._seed,
                config_name=config_name,  # Pass as config_name to HF Dataset
                **self._config,
            )

            # Import service components
            from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
            from ember.core.utils.data.base.samplers import DatasetSampler
            from ember.core.utils.data.base.validators import DatasetValidator
            from ember.core.utils.data.service import DatasetService

            # Create data service with transformers
            service = DatasetService(
                loader=HuggingFaceDatasetLoader(),
                validator=DatasetValidator(),
                sampler=DatasetSampler(),
                transformers=self._transformers,
            )

            # Load and prepare dataset
            entries = service.load_and_prepare(
                dataset_info=dataset_entry.info,
                prepper=dataset_entry.prepper,
                config=config,
                num_samples=self._sample_size,
            )

            return Dataset(entries=entries, info=dataset_entry.info)


class DataAPI:
    """Unified API for data operations.

    Provides a clean facade for all data operations with explicit
    dependency management through context.
    """

    # Import registry at class level to make it available for backward compatibility
    from ember.core.utils.data.registry import DATASET_REGISTRY

    def __init__(self, context: Union[EmberContext, DataContext]):
        """Initialize with explicit context.

        Args:
            context: EmberContext or DataContext, uses default if not provided
        """
        if isinstance(context, EmberContext):
            # Store EmberContext and get DataContext from it
            self._ember_context = context
            self._data_context = DataContext.create_from_ember_context(context)
        else:
            # Use provided DataContext
            self._data_context = context
            self._ember_context = None

    def initialize_registry(self) -> None:
        """Initialize registry for backward compatibility.

        Required by tests that call this method directly.

        This is a no-op since the registry is initialized during context creation.
        """
        # Registry is already initialized by data context
        pass

    def __call__(
        self,
        dataset_name: str,
        *,
        streaming: bool = True,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[Iterator[DataItem], List[DatasetEntry]]:
        """Load dataset with specified parameters.

        Args:
            dataset_name: Name of dataset to load
            streaming: Whether to use streaming mode
            limit: Optional maximum number of items
            **kwargs: Additional filtering criteria

        Returns:
            Dataset items or iterator

        Raises:
            ValueError: If dataset not found
        """
        # Start with builder
        builder = self.builder().from_registry(dataset_name)

        # Apply limit if specified
        if limit is not None:
            builder.limit(limit)

        # Set streaming mode
        builder.streaming(streaming)

        # Apply any additional filters
        if kwargs:

            def filter_fn(item: Dict[str, Any]) -> bool:
                for key, value in kwargs.items():
                    if item.get(key) != value:
                        return False
                return True

            builder.transform(lambda item: item if filter_fn(item) else None)

        # Build dataset
        return builder.build()

    def builder(self) -> DatasetBuilder:
        """Create dataset builder with current context.

        Returns:
            Dataset builder
        """
        if self._ember_context:
            return DatasetBuilder(context=self._ember_context)
        return DatasetBuilder(context=self._data_context)

    def list(self) -> List[str]:
        """List all available datasets.

        Returns:
            List of dataset names
        """
        return self._data_context.registry.list_datasets()

    def info(self, name: str) -> Optional[DatasetInfo]:
        """Get information about a dataset.

        Args:
            name: Dataset name

        Returns:
            Dataset information or None if not found
        """
        dataset = self._data_context.registry.get(name=name)
        if dataset and hasattr(dataset, "info"):
            return dataset.info
        return None

    def register(
        self,
        *,
        name: str,
        source: str,
        task_type: Union[TaskType, str],
        prepper_class: Optional[Any] = None,
        description: str = "",
    ) -> None:
        """Register a new dataset.

        Args:
            name: Dataset name
            source: Dataset source
            task_type: Task type
            prepper_class: Optional prepper class
            description: Optional description
        """
        # Convert string task_type to enum if needed
        if isinstance(task_type, str):
            task_enum = getattr(TaskType, task_type.upper(), None)
            if task_enum is None:
                raise ValueError(f"Invalid task type: {task_type}")
            task_type = task_enum

        # Use the DataContext's register_dataset method
        self._data_context.register_dataset(
            name=name,
            source=source,
            task_type=task_type,
            prepper_class=prepper_class,
            description=description,
        )


__all__ = [
    # Primary API
    "DataAPI",
    "Dataset",
    "DatasetBuilder",
    "DatasetConfig",
    # Data models
    "DataItem",
    "DatasetInfo",
    "DatasetEntry",
    "TaskType",
    # Context integration
    "DataContext",
]
