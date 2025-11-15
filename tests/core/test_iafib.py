import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
import polars as pl
import numpy as np

from pyhealth.datasets.iafib import _download_data_to_path, iAFibDataset


class TestDownloadDataToPath:
    """Tests for _download_data_to_path function"""

    def test_download_success_with_content_disposition(self):
        """Test successful download with Content-Disposition header"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Disposition": 'attachment; filename="data.zip"'}
            mock_response.content = b"zip content"
            mock_response.iter_content = lambda chunk_size: [b"zip content"]

            with patch("requests.get", return_value=mock_response), \
                 patch("builtins.open", mock_open()), \
                 patch("zipfile.ZipFile") as mock_zip, \
                 patch("os.listdir", return_value=["extracted_folder"]):
                
                result = _download_data_to_path("http://example.com/data.zip", tmp_dir)
                
                assert result == "extracted_folder"
                mock_zip.return_value.__enter__.return_value.extractall.assert_called_once()

    def test_download_success_without_content_disposition(self):
        """Test successful download without Content-Disposition header"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.content = b"zip content"
            mock_response.iter_content = lambda chunk_size: [b"zip content"]

            with patch("requests.get", return_value=mock_response), \
                 patch("builtins.open", mock_open()), \
                 patch("zipfile.ZipFile") as mock_zip, \
                 patch("os.listdir", return_value=["extracted_folder"]):
                
                result = _download_data_to_path("http://example.com/data.zip", tmp_dir)
                
                assert result == "extracted_folder"

    def test_download_failure_status_code(self):
        """Test download failure with non-200 status code"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_response = Mock()
            mock_response.status_code = 404

            with patch("requests.get", return_value=mock_response):
                with pytest.raises(Exception, match="Failed to download data"):
                    _download_data_to_path("http://example.com/data.zip", tmp_dir)

    def test_download_network_error(self):
        """Test download failure due to network error"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("requests.get", side_effect=Exception("Network error")):
                with pytest.raises(Exception, match="Failed to download data"):
                    _download_data_to_path("http://example.com/data.zip", tmp_dir)

    def test_extract_failure(self):
        """Test failure during zip extraction"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.content = b"invalid zip content"
            mock_response.iter_content = lambda chunk_size: [b"invalid zip content"]

            with patch("requests.get", return_value=mock_response), \
                 patch("builtins.open", mock_open()), \
                 patch("zipfile.ZipFile", side_effect=Exception("Bad zip file")):
                
                with pytest.raises(Exception, match="Failed to extract zip file"):
                    _download_data_to_path("http://example.com/data.zip", tmp_dir)

    def test_creates_directory_if_not_exists(self):
        """Test that save directory is created if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "new_dir")
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.content = b"content"
            mock_response.iter_content = lambda chunk_size: [b"content"]

            with patch("requests.get", return_value=mock_response), \
                 patch("builtins.open", mock_open()), \
                 patch("zipfile.ZipFile") as mock_zip, \
                 patch("os.listdir", return_value=["extracted"]):
                
                result = _download_data_to_path("http://example.com/data.zip", save_path)
                assert os.path.exists(save_path)


class TestiAFibDataset:
    """Tests for iAFibDataset class"""

    @patch("pyhealth.datasets.iafib.BaseDataset.__init__", return_value=None)
    def test_init_creates_root_directory(self, mock_base_init):
        """Test that __init__ creates root directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, "dataset_root")
            
            dataset = iAFibDataset(root=root)
            
            assert os.path.exists(root)
            assert dataset.extract_subdir == "extracted"
            assert isinstance(dataset.arrays, dict)

    @patch("pyhealth.datasets.iafib.BaseDataset.__init__", return_value=None)
    def test_init_with_custom_extract_subdir(self, mock_base_init):
        """Test initialization with custom extract_subdir"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = iAFibDataset(root=tmp_dir, extract_subdir="custom_dir")
            
            assert dataset.extract_subdir == "custom_dir"

    @patch("pyhealth.datasets.iafib.BaseDataset.__init__", return_value=None)
    def test_init_calls_parent_with_correct_args(self, mock_base_init):
        """Test that __init__ calls parent class with correct arguments"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_name = "test_dataset"
            config_path = "/path/to/config"
            
            dataset = iAFibDataset(
                root=tmp_dir,
                dataset_name=dataset_name,
                config_path=config_path,
                dev=True
            )
            
            mock_base_init.assert_called_once()
            call_kwargs = mock_base_init.call_args[1]
            assert call_kwargs["tables"] == ["iAFib"]
            assert call_kwargs["dataset_name"] == dataset_name
            assert call_kwargs["config_path"] == config_path
            assert call_kwargs["dev"] is True

    @patch("pyhealth.datasets.iafib.iAFibDataset.load_data", return_value=pl.LazyFrame({"col": [1, 2, 3]}))
    @patch("pyhealth.datasets.iafib.BaseDataset.__init__", return_value=None)
    def test_load_data_returns_lazyframe(self, mock_base_init, mock_load_data):
        """Test that load_data returns a LazyFrame"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = iAFibDataset(root=tmp_dir)
            result = dataset.load_data()
            
            assert isinstance(result, pl.LazyFrame)

    @patch("pyhealth.datasets.iafib._download_data_to_path")
    @patch("pyhealth.datasets.iafib.wfdb.rdrecord")
    @patch("pyhealth.datasets.iafib.BaseDataset.__init__", return_value=None)
    def test_load_data_processes_qrs_files(self, mock_base_init, mock_rdrecord, mock_download):
        """Test that load_data correctly processes QRS files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup mocks
            mock_download.return_value = "test_folder"
            mock_record = Mock()
            mock_record.record_name = "patient001_region1"
            mock_record.sig_name = ["CS1", "CS2", "OTHER"]
            mock_record.p_signal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            mock_record.fs = 1000
            mock_record.comments = ["<AF>: Yes", "<noise>: low"]
            mock_rdrecord.return_value = mock_record

            # Mock os.listdir to return test files
            with patch("os.listdir", side_effect=[["patient001_region1_qrs.dat"]]):
                dataset = iAFibDataset(root=tmp_dir)
                dataset.root = tmp_dir
                dataset.extract_subdir = "extracted"
                dataset.config = Mock()
                dataset.config.tables = {"iAFib": Mock(source_url="http://test.com")}
                dataset.tables = ["iAFib"]
                
                result = dataset.load_data()
                
                assert result is not None
                assert isinstance(result, pl.LazyFrame)
                mock_download.assert_called_once()
                mock_rdrecord.assert_called()

    @patch("pyhealth.datasets.iafib._download_data_to_path")
    @patch("pyhealth.datasets.iafib.wfdb.rdrecord")
    @patch("pyhealth.datasets.iafib.BaseDataset.__init__", return_value=None)
    def test_load_data_concatenates_multiple_records(self, mock_base_init, mock_rdrecord, mock_download):
        """Test that load_data concatenates multiple QRS records"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_download.return_value = "test_folder"
            
            # Create mock records
            mock_record1 = Mock()
            mock_record1.record_name = "patient001_region1"
            mock_record1.sig_name = ["CS1"]
            mock_record1.p_signal = np.array([[1, 2, 3]])
            mock_record1.fs = 1000
            mock_record1.comments = []
            
            mock_record2 = Mock()
            mock_record2.record_name = "patient002_region1"
            mock_record2.sig_name = ["CS1"]
            mock_record2.p_signal = np.array([[4, 5, 6]])
            mock_record2.fs = 1000
            mock_record2.comments = []
            
            mock_rdrecord.side_effect = [mock_record1, mock_record2]

            with patch("os.listdir", side_effect=[["file1_qrs.dat", "file2_qrs.dat"]]):
                dataset = iAFibDataset(root=tmp_dir)
                dataset.root = tmp_dir
                dataset.extract_subdir = "extracted"
                dataset.config = Mock()
                dataset.config.tables = {"iAFib": Mock(source_url="http://test.com")}
                dataset.tables = ["iAFib"]
                
                result = dataset.load_data()
                
                assert result is not None
                assert mock_rdrecord.call_count == 2

    @patch("pyhealth.datasets.iafib._download_data_to_path")
    @patch("pyhealth.datasets.iafib.BaseDataset.__init__", return_value=None)
    def test_load_data_skips_non_qrs_files(self, mock_base_init, mock_download):
        """Test that load_data skips files without 'qrs' in the name"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_download.return_value = "test_folder"

            with patch("os.listdir", side_effect=[["file1.txt", "file2.csv", "file3_info.dat"]]):
                dataset = iAFibDataset(root=tmp_dir)
                dataset.root = tmp_dir
                dataset.extract_subdir = "extracted"
                dataset.config = Mock()
                dataset.config.tables = {"iAFib": Mock(source_url="http://test.com")}
                dataset.tables = ["iAFib"]
                
                result = dataset.load_data()
                
                # Should return None since no QRS files were processed
                assert result is None
