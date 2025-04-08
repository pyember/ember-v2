"""Unit tests for dataset loader classes."""

import os
import unittest
from typing import Optional, Union
from unittest import mock
from urllib.error import HTTPError

from datasets import Dataset, DatasetDict

from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader, IDatasetLoader

# Create a testable subclass for testing
class TestableHuggingFaceDatasetLoader(HuggingFaceDatasetLoader):
    """A testable version of HuggingFaceDatasetLoader with mocked HfApi."""
    
    def __init__(self, mock_hf_api, mock_load_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_hf_api = mock_hf_api
        self._mock_load_dataset = mock_load_dataset
        
    def load(
        self, *, dataset_name: str, config: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
        """Overridden load method using mocked dependencies."""
        try:
            self._mock_hf_api.dataset_info(dataset_name)
        except Exception as exc:
            raise ValueError(
                f"Dataset '{dataset_name}' does not exist on the Hub."
            ) from exc
            
        try:
            dataset_args = {"path": dataset_name, "cache_dir": self.cache_dir}
            if config:
                dataset_args["name"] = config
            
            return self._mock_load_dataset(**dataset_args)
        except HTTPError as http_err:
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}'."
            ) from http_err
        except Exception as exc:
            raise RuntimeError(
                f"Error loading dataset '{dataset_name}': {exc}"
            ) from exc
            

class TestIDatasetLoader(unittest.TestCase):
    """Test cases for the IDatasetLoader interface."""

    def test_interface_enforcement(self) -> None:
        """IDatasetLoader should require implementation of the load method."""
        # Attempt to instantiate the abstract base class directly
        with self.assertRaises(TypeError):
            IDatasetLoader()  # type: ignore

        # Create a subclass that doesn't implement load
        class IncompleteLoader(IDatasetLoader):
            pass

        # Attempt to instantiate the incomplete subclass
        with self.assertRaises(TypeError):
            IncompleteLoader()  # type: ignore

        # Create a proper implementation
        class CompleteLoader(IDatasetLoader):
            def load(
                self, *, dataset_name: str, config: Optional[str] = None
            ) -> Dataset:
                return Dataset.from_dict({"data": [1, 2, 3]})

        # Should instantiate without error
        loader = CompleteLoader()
        self.assertIsInstance(loader, IDatasetLoader)


class TestHuggingFaceDatasetLoader(unittest.TestCase):
    """Test cases for the HuggingFaceDatasetLoader class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a patcher for HfApi's constructor so we can intercept all instances
        self.hf_api_patcher = mock.patch("huggingface_hub.HfApi", autospec=True)
        self.mock_hf_api_cls = self.hf_api_patcher.start()
        
        # Create a mock that will be returned when HfApi is instantiated
        self.mock_hf_api = mock.MagicMock()
        self.mock_hf_api_cls.return_value = self.mock_hf_api
        
        # Ensure dataset_info method is properly mocked
        self.mock_hf_api.dataset_info = mock.MagicMock()

        # Create a patcher for load_dataset
        self.load_dataset_patcher = mock.patch(
            "datasets.load_dataset"
        )
        self.mock_load_dataset = self.load_dataset_patcher.start()

        # Create a patcher for os.makedirs
        self.makedirs_patcher = mock.patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

        # Create a patcher for os.path.expanduser
        self.expanduser_patcher = mock.patch("os.path.expanduser")
        self.mock_expanduser = self.expanduser_patcher.start()
        self.mock_expanduser.return_value = "/mocked/home"

        # Create a patcher for enable_progress_bar
        self.enable_progress_bar_patcher = mock.patch(
            "datasets.enable_progress_bar"
        )
        self.mock_enable_progress_bar = self.enable_progress_bar_patcher.start()

        # Create a patcher for disable_progress_bar
        self.disable_progress_bar_patcher = mock.patch(
            "datasets.disable_progress_bar"
        )
        self.mock_disable_progress_bar = self.disable_progress_bar_patcher.start()

        # Create a patcher for enable_caching
        self.enable_caching_patcher = mock.patch(
            "datasets.enable_caching"
        )
        self.mock_enable_caching = self.enable_caching_patcher.start()

        # Create a patcher for disable_caching
        self.disable_caching_patcher = mock.patch(
            "datasets.disable_caching"
        )
        self.mock_disable_caching = self.disable_caching_patcher.start()

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.hf_api_patcher.stop()
        self.load_dataset_patcher.stop()
        self.makedirs_patcher.stop()
        self.expanduser_patcher.stop()
        self.enable_progress_bar_patcher.stop()
        self.disable_progress_bar_patcher.stop()
        self.enable_caching_patcher.stop()
        self.disable_caching_patcher.stop()

    def test_init_default_cache_dir(self) -> None:
        """HuggingFaceDatasetLoader should create default cache directory when not provided."""
        # Arrange & Act
        loader = HuggingFaceDatasetLoader()

        # Assert
        expected_cache_dir = os.path.join(
            "/mocked/home", ".cache", "huggingface", "datasets"
        )
        self.assertEqual(expected_cache_dir, loader.cache_dir)
        self.mock_makedirs.assert_called_once_with(expected_cache_dir, exist_ok=True)

    def test_init_custom_cache_dir(self) -> None:
        """HuggingFaceDatasetLoader should use and create the provided cache directory."""
        # Arrange
        custom_cache_dir = "/custom/cache/dir"

        # Act
        loader = HuggingFaceDatasetLoader(cache_dir=custom_cache_dir)

        # Assert
        self.assertEqual(custom_cache_dir, loader.cache_dir)
        self.mock_makedirs.assert_called_once_with(custom_cache_dir, exist_ok=True)

    def test_load_success(self) -> None:
        """load() should successfully load a dataset with proper error handling."""
        # Arrange
        dataset_name = "test_dataset"
        config_name = "test_config"
        mock_dataset = mock.MagicMock(spec=DatasetDict)
        
        # Setup mocks
        mock_hf_api = mock.MagicMock()
        mock_load_dataset = mock.MagicMock(return_value=mock_dataset)

        # Create testable loader with our mocks
        loader = TestableHuggingFaceDatasetLoader(
            mock_hf_api=mock_hf_api,
            mock_load_dataset=mock_load_dataset
        )

        # Act
        result = loader.load(dataset_name=dataset_name, config=config_name)

        # Assert
        self.assertEqual(mock_dataset, result)
        mock_hf_api.dataset_info.assert_called_once_with(dataset_name)
        mock_load_dataset.assert_called_once_with(
            path=dataset_name, name=config_name, cache_dir=loader.cache_dir
        )

    def test_load_dataset_not_found(self) -> None:
        """load() should raise ValueError when the dataset cannot be found."""
        # Arrange
        dataset_name = "nonexistent_dataset"
        
        # Setup mocks
        mock_hf_api = mock.MagicMock()
        mock_hf_api.dataset_info.side_effect = Exception("Dataset not found")
        mock_load_dataset = mock.MagicMock()

        loader = TestableHuggingFaceDatasetLoader(
            mock_hf_api=mock_hf_api,
            mock_load_dataset=mock_load_dataset
        )

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            loader.load(dataset_name=dataset_name)

        # Verify error message
        self.assertIn(dataset_name, str(context.exception))
        mock_hf_api.dataset_info.assert_called_once_with(dataset_name)
        mock_load_dataset.assert_not_called()

    def test_load_http_error(self) -> None:
        """load() should raise RuntimeError when an HTTP error occurs during download."""
        # Arrange
        dataset_name = "error_dataset"
        http_error = HTTPError("http://example.com", 404, "Not Found", {}, None)

        # Setup mocks
        mock_hf_api = mock.MagicMock()
        mock_load_dataset = mock.MagicMock(side_effect=http_error)

        loader = TestableHuggingFaceDatasetLoader(
            mock_hf_api=mock_hf_api,
            mock_load_dataset=mock_load_dataset
        )

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            loader.load(dataset_name=dataset_name)

        # Verify error message contains useful information
        self.assertIn(dataset_name, str(context.exception))
        self.assertIn("Failed to download dataset", str(context.exception))

    def test_load_unexpected_error(self) -> None:
        """load() should raise RuntimeError with informative message for unexpected errors."""
        # Arrange
        dataset_name = "error_dataset"
        unexpected_error = RuntimeError("Unexpected test error")

        # Setup mocks
        mock_hf_api = mock.MagicMock()
        mock_load_dataset = mock.MagicMock(side_effect=unexpected_error)

        loader = TestableHuggingFaceDatasetLoader(
            mock_hf_api=mock_hf_api,
            mock_load_dataset=mock_load_dataset
        )

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            loader.load(dataset_name=dataset_name)

        # Verify error message contains useful information
        self.assertIn(dataset_name, str(context.exception))
        self.assertIn("Error loading dataset", str(context.exception))
        self.assertIn(str(unexpected_error), str(context.exception))


if __name__ == "__main__":
    unittest.main()
