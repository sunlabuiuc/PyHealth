import sys
import types
import copy


def _install_fake_litdata() -> None:
    if "litdata" in sys.modules:
        return

    litdata = types.ModuleType("litdata")

    class StreamingDataset:
        def __init__(self, *args, **kwargs):
            self.data = []

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    litdata.StreamingDataset = StreamingDataset

    # litdata.streaming.item_loader
    streaming = types.ModuleType("litdata.streaming")
    item_loader = types.ModuleType("litdata.streaming.item_loader")

    class ParquetLoader:
        pass

    class TokensLoader:
        pass

    item_loader.ParquetLoader = ParquetLoader
    item_loader.TokensLoader = TokensLoader

    # litdata.streaming.writer
    writer = types.ModuleType("litdata.streaming.writer")

    class BinaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_item(self, *args, **kwargs):
            return None

        def done(self):
            return None

    writer.BinaryWriter = BinaryWriter
    writer._INDEX_FILENAME = "index.json"

    # litdata.processing.data_processor
    processing = types.ModuleType("litdata.processing")
    data_processor = types.ModuleType("litdata.processing.data_processor")

    def in_notebook():
        return False

    data_processor.in_notebook = in_notebook

    # litdata.utilities.train_test_split
    utilities = types.ModuleType("litdata.utilities")
    train_test_split = types.ModuleType("litdata.utilities.train_test_split")

    def deepcopy_dataset(dataset):
        return copy.deepcopy(dataset)

    train_test_split.deepcopy_dataset = deepcopy_dataset

    # Register all modules
    sys.modules["litdata"] = litdata
    sys.modules["litdata.streaming"] = streaming
    sys.modules["litdata.streaming.item_loader"] = item_loader
    sys.modules["litdata.streaming.writer"] = writer
    sys.modules["litdata.processing"] = processing
    sys.modules["litdata.processing.data_processor"] = data_processor
    sys.modules["litdata.utilities"] = utilities
    sys.modules["litdata.utilities.train_test_split"] = train_test_split


_install_fake_litdata()
