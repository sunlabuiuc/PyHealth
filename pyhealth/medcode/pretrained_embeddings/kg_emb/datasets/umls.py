import logging
import os
import shutil
from pathlib import Path

from pyhealth.datasets.base_dataset import is_url

from .base_kg_dataset import BaseKGDataset

logger = logging.getLogger(__name__)

RAW_FILE = "graph.txt"
PREPARED_FILE = "umls-pyhealth.tsv"
HEADER = b"head\trelation\ttail\n"


class UMLSDataset(BaseKGDataset):
    """UMLS knowledge graph dataset.

    Dataset is available at https://www.nlm.nih.gov/research/umls/index.html

    ``root`` holds ``graph.txt``: one triple per line, ``head``, ``relation``
    and ``tail`` separated by tabs, without a header. On first use,
    :meth:`prepare_metadata` writes ``umls-pyhealth.tsv`` next to it, the same
    rows under a ``head``/``relation``/``tail`` header, which the bundled
    ``configs/umls.yaml`` reads. ``graph.txt`` itself is never modified. The
    copy is made again whenever ``graph.txt`` is newer than it or differs from
    it in size, so replacing ``graph.txt`` gives a new dataset cache.

    Entity and relation ids are assigned by sorted name (see
    :class:`BaseKGDataset`), no longer by order of first appearance as in
    PyHealth 1.x: embeddings trained with 1.x must be mapped through the
    ``id2entity`` / ``id2relation`` saved with them.

    Args:
        root: Local directory holding ``graph.txt``. URLs are not accepted;
            the 1.x bucket ``https://storage.googleapis.com/pyhealth/umls/``
            serves ``graph.txt``, which can be downloaded into ``root``.
        dataset_name: Name of the dataset. Defaults to ``"umls"``.
        config_path: Config to use instead of the bundled ``umls.yaml``.
        **kwargs: Passed to :class:`BaseKGDataset` (``cache_dir``,
            ``num_workers``, ``dev``).

    Raises:
        ValueError: If ``root`` is a URL.
        FileNotFoundError: If neither ``graph.txt`` nor
            ``umls-pyhealth.tsv`` is in ``root``.

    Examples:
        >>> from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import (
        ...     BaseKGDataset,
        ...     UMLSDataset,
        ... )
        >>> issubclass(UMLSDataset, BaseKGDataset)
        True
        >>> dataset = UMLSDataset(root="/path/to/umls")  # doctest: +SKIP
        >>> dataset.num_entities  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        if is_url(root):
            # The public bucket used by PyHealth 1.x serves graph.txt only,
            # and the prepared copy cannot be written next to a URL.
            raise ValueError(
                f"UMLSDataset needs a local root; download {root.rstrip('/')}/"
                f"{RAW_FILE} into a directory and pass that directory."
            )
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "umls.yaml"
        if self._needs_preparation(root):
            self.prepare_metadata(root)
        super().__init__(
            root=root,
            config_path=config_path,
            dataset_name=dataset_name or "umls",
            **kwargs,
        )

    @staticmethod
    def _needs_preparation(root: str) -> bool:
        """Whether ``umls-pyhealth.tsv`` is missing or older than ``graph.txt``.

        The dataset cache is keyed on the prepared copy, so a ``graph.txt``
        replaced after the copy was made must trigger a new copy; otherwise
        the old graph would be reused without notice.
        """
        source = os.path.join(root, RAW_FILE)
        target = os.path.join(root, PREPARED_FILE)
        if not os.path.exists(target):
            return True
        if not os.path.exists(source):
            # Only the prepared copy was provided; use it as it is.
            return False
        src, dst = os.stat(source), os.stat(target)
        return (
            dst.st_size != src.st_size + len(HEADER)
            or src.st_mtime_ns > dst.st_mtime_ns
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Writes ``umls-pyhealth.tsv``: ``graph.txt`` under a header line.

        The rows are copied byte for byte; only the header
        ``head\\trelation\\ttail`` is added, because the table loader reads the
        column names from the first line.

        Args:
            root: Directory holding ``graph.txt``.

        Raises:
            FileNotFoundError: If ``graph.txt`` is not in ``root``.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> root = Path(tempfile.mkdtemp())
            >>> _ = (root / "graph.txt").write_text("C1\\tPAR\\tC2\\n")
            >>> UMLSDataset.prepare_metadata(str(root))
            >>> (root / "umls-pyhealth.tsv").read_text()
            'head\\trelation\\ttail\\nC1\\tPAR\\tC2\\n'
        """
        source = os.path.join(root, RAW_FILE)
        if not os.path.exists(source):
            raise FileNotFoundError(f"UMLS graph not found: {source}")
        target = os.path.join(root, PREPARED_FILE)
        logger.info(f"Writing {target} from {source}")
        # Written under a temporary name and renamed when complete: an
        # interrupted copy must not leave a truncated file that later runs
        # would take as prepared.
        partial = target + ".partial"
        with open(source, "rb") as src, open(partial, "wb") as dst:
            dst.write(HEADER)
            shutil.copyfileobj(src, dst)
        os.replace(partial, target)


if __name__ == "__main__":
    dataset = UMLSDataset(root="/path/to/umls", dev=True)
    dataset.stats()
    print(dataset.num_entities, dataset.num_relations)
