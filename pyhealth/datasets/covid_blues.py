"""COVID-BLUES bluepoint lung ultrasound dataset."""

import logging
import os
import re
import shutil
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

logger = logging.getLogger(__name__)

_VIDEO_RE = re.compile(r"^patient_(\d+)_([LR][123])(?:_(\d+))?$")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_METADATA_FILE = "covid-blues-metadata-pyhealth.csv"
_MARKER_FILE = "covid-blues-metadata-key.txt"
_COLUMNS = [
    "patient_id", "record_id", "video_id", "blue_point", "repeat",
    "frame_index", "image_path", "covid", "severity_score",
]


def _video_stem(name: str) -> str:
    """Return ``name`` without a trailing ``.mp4`` (case-insensitive)."""
    name = name.strip()
    return name[:-4] if name.lower().endswith(".mp4") else name


def _decode_video(video_path: Path, out_dir: Path) -> int:
    """Write every frame of a video as a PNG into ``out_dir``.

    Uses PyAV (``av``) when installed, else OpenCV (``cv2``). Both are
    optional and imported here only.

    Returns:
        The number of frames written.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        with av.open(str(video_path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                frame.to_image().save(out_dir / f"{index:06d}.png")
                written += 1
        return written
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Decoding COVID-BLUES videos needs the optional package av "
            "(`pip install av`) or opencv-python (`pip install "
            "opencv-python`). Alternatively, extract the frames yourself "
            "(e.g. `ffmpeg -i patient_1_L1.mp4 frames/patient_1_L1/%06d.png`) "
            "and pass frames_dir."
        ) from exc
    capture = cv2.VideoCapture(str(video_path))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(out_dir / f"{index:06d}.png")
            written += 1
            index += 1
    finally:
        capture.release()
    return written


class COVIDBLUESDataset(BaseDataset):
    """COVID-BLUES: bluepoint lung ultrasound videos of COVID-19 suspects.

    63 patients (33 PCR-positive, 30 negative) were scanned at up to six BLUE
    points (L1-L3, R1-R3), giving 362 videos of 7-101 frames each, most of
    them 60-101 (31,746 frames in all, counted by decoding with PyAV; the
    dataset README says 60-100). This loader works on frames: there is one event per frame
    image, with the patient's PCR result (``cov_test``) as ``covid``, the
    video id, BLUE point and frame index. When ``severity.csv`` is present,
    each frame also gets its video's expert ``severity_score`` (0-3).

    Frames: the primary input is a folder of pre-extracted frames,
    ``frames_dir/<video stem>/<frame>.png`` (or ``.jpg``), for example from
    ``ffmpeg -i lus_videos/patient_1_L1.mp4 frames/patient_1_L1/%06d.png``.
    With ``decode_videos=True`` the loader instead decodes the ``.mp4`` files
    itself with the optional ``av`` (PyAV) or ``opencv-python`` package and
    writes PNG frames to ``frames_dir`` (by default a folder in the PyHealth
    cache, never ``root``); all frames are written. ``frame_step`` keeps
    every n-th frame of each video (in sorted file order) in both cases.

    Frames of one patient are highly correlated, so always split by
    ``patient_id`` (for example with :class:`~pyhealth.datasets.PatientSplit`),
    and consider aggregating predictions per video or patient.

    Differences from CLIMB: CLIMB also classifies frames by PCR result and
    splits on the first 75% of patients sorted by id. Its split is
    patient-level too, but fixed by id order; this loader leaves the split
    to the user (a random :class:`~pyhealth.datasets.PatientSplit` is
    recommended). CLIMB's age-regression question set is not included.

    Getting the data:
        - Source: https://github.com/NinaWie/COVID-BLUES (mirrored on
          Hugging Face, https://huggingface.co/datasets/jannisborn/COVID-BLUES).
        - Access: open, no account, registration or agreement needed.
        - Download: the whole repository, about 70 MB (362 ``.mp4`` videos
          plus CSVs, no Git LFS)::

              git clone https://github.com/NinaWie/COVID-BLUES

          or download
          https://github.com/NinaWie/COVID-BLUES/archive/refs/heads/main.zip
          and ``unzip`` it (it unpacks to ``COVID-BLUES-main/``; use that
          folder as ``root``).
        - Frames: either pass ``decode_videos=True`` (needs ``pip install
          av`` or ``pip install opencv-python``), or extract them yourself
          with ffmpeg, one folder per video stem::

              cd COVID-BLUES
              for f in lus_videos/*.mp4; do
                  s=$(basename "$f" .mp4); mkdir -p ../frames/$s
                  ffmpeg -loglevel error -i "$f" ../frames/$s/%06d.png
              done

        - ``root`` and ``frames_dir`` layout::

              COVID-BLUES/                     # root
                  clinical_variables.csv       # patient_id, cov_test, ...
                  severity.csv                 # optional; video_file, Severity Score
                  lus_videos/
                      patient_1_L1.mp4
                      patient_15_L2_2.mp4      # _2: repeated recording
                      ...
              frames/                          # frames_dir
                  patient_1_L1/
                      000001.png
                      ...

          Frame folders must be named after the video stems
          (``patient_<id>_<L|R><1-3>[_<n>]``). ``severity.csv``'s
          ``video_file`` values are matched to those stems (a trailing
          ``.mp4`` is ignored); if none match, a warning is logged and
          ``severity_score`` stays empty.

    Licence: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
    (CC BY-NC-ND 4.0). Non-commercial use only, and adapted material (which
    may include extracted frames) must not be shared. Keep extracted frames
    local; do not redistribute them.

    Citation:
        Wiedemann, N., de Korte-de Boer, D., Richter, M., van de Weijer, S.,
        Buhre, C., Eggert, F. A. M., Aarnoudse, S., Grevendonk, L., Röber,
        S., Remie, C. M. E., Buhre, W., Henry, R., & Born, J. (2025).
        COVID-BLUeS - A prospective study on the value of AI in lung
        ultrasound analysis. IEEE Journal of Biomedical and Health
        Informatics, 29(9), 6301-6310. https://doi.org/10.1109/JBHI.2025.3543686

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Args:
        root: The COVID-BLUES clone.
        frames_dir: Folder of pre-extracted frames, one sub-folder per video
            stem. With ``decode_videos=True`` this is where decoded frames are
            written; it then defaults to ``<cache>/metadata/<dataset_name>/
            frames``.
        decode_videos: Decode the ``.mp4`` files that have no frames yet
            (needs ``av`` or ``opencv-python``). Default False.
        frame_step: Keep every ``frame_step``-th frame. Default 1 (all).
        dataset_name: Optional dataset name. Default ``"covid_blues"``.
        config_path: Optional table configuration path.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the derived metadata CSV. Defaults to
            ``<cache>/metadata/<dataset_name>``.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Regenerate the metadata CSV, e.g. after adding
            frames.

    Examples:
        >>> from pyhealth.datasets import COVIDBLUESDataset
        >>> dataset = COVIDBLUESDataset(  # doctest: +SKIP
        ...     root="/path/to/COVID-BLUES", frames_dir="/path/to/frames"
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "covid_blues"
    default_table = "frames"

    def __init__(
        self,
        root: str,
        frames_dir: str | None = None,
        decode_videos: bool = False,
        frame_step: int = 1,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        self.source_root = Path(root).expanduser().resolve()
        if int(frame_step) < 1:
            raise ValueError(f"frame_step must be >= 1, got {frame_step}.")
        self.frame_step = int(frame_step)
        self.decode_videos = bool(decode_videos)
        self.clinical_path = self.source_root / "clinical_variables.csv"
        if not self.clinical_path.exists():
            raise FileNotFoundError(
                f"clinical_variables.csv not found under {self.source_root}. "
                "Clone https://github.com/NinaWie/COVID-BLUES."
            )

        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if frames_dir is not None:
            self.frames_dir = Path(frames_dir).expanduser().resolve()
        elif self.decode_videos:
            self.frames_dir = self.metadata_dir / "frames"
        else:
            raise ValueError(
                "Pass frames_dir (a folder of pre-extracted frames, one "
                "sub-folder per video) or decode_videos=True."
            )
        self.metadata_path = self.metadata_dir / _METADATA_FILE
        marker = self.metadata_dir / _MARKER_FILE
        key = (
            f"root={self.source_root}|frames={self.frames_dir}"
            f"|frame_step={self.frame_step}"
        )
        with FileLock(str(self.metadata_dir / (_METADATA_FILE + ".lock"))):
            if self.decode_videos:
                self.extract_frames()
            current = marker.read_text().strip() if marker.exists() else None
            if (
                refresh_metadata
                or self.decode_videos
                or not self.metadata_path.exists()
                or current != key
            ):
                self.prepare_metadata()
                marker.write_text(key)

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "covid_blues.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def extract_frames(self) -> None:
        """Decode the videos that have no frame folder yet into ``frames_dir``.

        Needs the optional ``av`` (PyAV) or ``opencv-python`` package. A
        video's folder is written under a temporary name and renamed when
        complete, so an interrupted run is redone next time. A video that
        yields no frames or makes the decoder raise (e.g. a corrupt file)
        gets no folder and a warning, so it is retried on the next run
        instead of being cached as empty; the other videos are still
        decoded. A missing decoder (``ImportError``) is still raised.

        Examples:
            >>> dataset = COVIDBLUESDataset(  # doctest: +SKIP
            ...     root="/path/to/COVID-BLUES", decode_videos=True
            ... )
            >>> dataset.extract_frames()  # doctest: +SKIP
        """
        videos = sorted((self.source_root / "lus_videos").glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(
                f"No .mp4 videos under {self.source_root / 'lus_videos'}."
            )
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        failed = []
        for video in videos:
            target = self.frames_dir / video.stem
            if target.is_dir():
                continue
            partial = self.frames_dir / f".{video.stem}.partial"
            if partial.exists():
                shutil.rmtree(partial)
            try:
                count = _decode_video(video, partial)
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001 - PyAV/cv2 raise many types
                shutil.rmtree(partial, ignore_errors=True)
                failed.append(video.name)
                logger.warning(
                    "Could not decode %s (%s: %s); it is left out and retried "
                    "on the next run.",
                    video,
                    type(exc).__name__,
                    exc,
                )
                continue
            if count == 0:
                shutil.rmtree(partial, ignore_errors=True)
                failed.append(video.name)
                logger.warning(
                    "Decoded no frames from %s; it is left out and retried "
                    "on the next run.",
                    video,
                )
                continue
            os.replace(partial, target)
            logger.info("Decoded %d frames from %s", count, video.name)
        if failed:
            logger.warning(
                "%d of %d COVID-BLUES videos gave no frames: %s",
                len(failed),
                len(videos),
                ", ".join(failed),
            )

    def prepare_metadata(self) -> None:
        """Write one row per frame image found in ``frames_dir``.

        Examples:
            >>> dataset = COVIDBLUESDataset(  # doctest: +SKIP
            ...     root="/path/to/COVID-BLUES", frames_dir="/path/to/frames"
            ... )
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        clinical = pd.read_csv(self.clinical_path)
        covid = {
            str(int(pid)): int(label)
            for pid, label in zip(clinical["patient_id"], clinical["cov_test"])
            if pd.notna(label)
        }
        severity = {}
        severity_path = self.source_root / "severity.csv"
        if severity_path.exists():
            table = pd.read_csv(severity_path)
            # Keys are video stems; tolerate a ``.mp4`` suffix.
            severity = {
                _video_stem(name): score
                for name, score in zip(
                    table["video_file"].astype(str), table["Severity Score"]
                )
            }
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"frames_dir not found: {self.frames_dir}")

        rows, skipped, scored = [], 0, 0
        for video_dir in sorted(p for p in self.frames_dir.iterdir() if p.is_dir()):
            match = _VIDEO_RE.match(video_dir.name)
            if match is None:
                continue
            patient, point, repeat = match.group(1), match.group(2), match.group(3)
            if patient not in covid:
                skipped += 1
                continue
            frames = sorted(
                f for f in video_dir.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES
            )
            score = severity.get(video_dir.name)
            scored += score is not None
            for position, frame in enumerate(frames):
                if position % self.frame_step:
                    continue
                digits = re.sub(r"\D", "", frame.stem)
                frame_index = int(digits) if digits else position
                rows.append(
                    {
                        "patient_id": patient,
                        "record_id": f"{video_dir.name}/{frame.stem}",
                        "video_id": video_dir.name,
                        "blue_point": point,
                        "repeat": int(repeat or 1),
                        "frame_index": frame_index,
                        "image_path": str(frame),
                        "covid": covid[patient],
                        "severity_score": "" if score is None or pd.isna(score) else score,
                    }
                )
        if not rows:
            raise FileNotFoundError(
                f"No frames found under {self.frames_dir}. Expected "
                "<frames_dir>/patient_<id>_<point>/<frame>.png."
            )
        if severity and not scored:
            logger.warning(
                "No severity.csv video_file value matches a frame folder name "
                "(expected video stems such as patient_1_L1); severity_score "
                "is empty for every frame."
            )
        tmp_path = self.metadata_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows, columns=_COLUMNS).to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.metadata_path)
        logger.info(
            "Wrote %d COVID-BLUES frames to %s (%d video folders without a "
            "clinical row)",
            len(rows),
            self.metadata_path,
            skipped,
        )

    @property
    def default_task(self):
        """Return the default frame-level COVID-19 classification task.

        Examples:
            >>> dataset = COVIDBLUESDataset(  # doctest: +SKIP
            ...     root="/path/to/COVID-BLUES", frames_dir="/path/to/frames"
            ... )
            >>> dataset.default_task.task_name  # doctest: +SKIP
            'COVIDBLUESFrameClassification'
        """
        from ..tasks.covid_blues_classification import COVIDBLUESFrameClassification

        return COVIDBLUESFrameClassification()
