"""Generate Synthea populations as CSV files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

import platformdirs

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SyntheaRelease:
    """Pinned Synthea release and the exporter keys this wrapper manages.

    Bumping Synthea means updating ``version``, ``url``, and ``sha256``
    together, then re-checking the exporter keys against the new release's
    ``synthea.properties``.

    Attributes:
        version (str): Synthea release tag.
        url (str): Download URL of the release's all-in-one JAR.
        sha256 (str): Expected SHA-256 digest of the JAR.
        base_directory_key (str): Property that sets the output root.
        csv_export_key (str): Property that enables CSV export.
        unsupported_export_flags (tuple[str, ...]): Exporter switches forced
            off so that only CSV output is produced.
        min_java_version (int): Oldest Java major version the JAR runs on.
    """

    version: str = "v4.0.0"
    url: str = (
        "https://github.com/synthetichealth/synthea/releases/download/"
        "v4.0.0/synthea-with-dependencies.jar"
    )
    sha256: str = "ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1"
    base_directory_key: str = "exporter.baseDirectory"
    csv_export_key: str = "exporter.csv.export"
    unsupported_export_flags: tuple[str, ...] = (
        "exporter.bfd.export",
        "exporter.ccda.export",
        "exporter.cdw.export",
        "exporter.clinical_note.export",
        "exporter.cpcds.export",
        "exporter.fhir.export",
        "exporter.fhir_dstu2.export",
        "exporter.fhir_stu3.export",
        "exporter.json.export",
        "exporter.symptoms.csv.export",
        "exporter.symptoms.text.export",
        "exporter.text.export",
    )
    min_java_version: int = 17

    def __post_init__(self) -> None:
        """Rejects a malformed pinned digest.

        Raises:
            ValueError: If ``sha256`` is not 64 lowercase hex characters.
        """
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(f"invalid Synthea SHA-256: {self.sha256!r}")


_RELEASE = _SyntheaRelease()

_JAVA_VERSION = re.compile(r'version "(\d+)(?:\.(\d+))?')
_PROPERTY_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_MANAGED_CONFIG_KEYS = frozenset(
    {
        _RELEASE.base_directory_key,
        _RELEASE.csv_export_key,
        *_RELEASE.unsupported_export_flags,
    }
)


def _parse_properties(text: str) -> dict[str, str | None]:
    """Parses Java properties text into a Python mapping.

    Commented properties are retained with a value of ``None`` so callers can
    discover properties that Synthea supports but leaves unset.

    Args:
        text (str): Contents of a Synthea properties file.

    Returns:
        dict[str, str | None]: Parsed property names and values.
    """
    config = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        disabled = line.startswith(("#", "!"))
        if disabled:
            line = line[1:].strip()
        match = re.match(r"([^=:\s]+)\s*(?:=|:)\s*(.*)", line)
        if not match:
            continue
        key, value = match.groups()
        if _PROPERTY_KEY.fullmatch(key):
            config[key] = None if disabled else value.strip()
    return config


def _optional_path(value: str | Path | None) -> Path | None:
    """Converts an optional path-like value to an expanded path.

    Args:
        value (str or Path, optional): Path-like value to convert.

    Returns:
        Path | None: The expanded path, or ``None`` when omitted.
    """
    return Path(value).expanduser() if value is not None else None


@dataclass(frozen=True)
class _GenerationSettings:
    """Validated settings describing one Synthea population.

    One instance is built per :meth:`Synthea.generate` call. Every field
    changes which patients Synthea produces, so all of them are part of the
    output fingerprint. Fields are documented on :meth:`Synthea.generate`.
    """

    population: int | None = None
    seed: int | None = None
    clinician_seed: int | None = None
    single_person_seed: int | None = None
    reference_date: str | None = None
    end_date: str | None = None
    gender: str | None = None
    age_range: str | None = None
    overflow_population: bool | None = None
    state: str | None = None
    city: str | None = None
    local_config_path: Path | None = None
    local_modules_dir: Path | None = None
    initial_population_snapshot_path: Path | None = None
    updated_population_snapshot_path: Path | None = None
    update_time_period: int | None = None
    fixed_record_path: Path | None = None
    keep_matching_patients_path: Path | None = None
    synthea_config: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    local_config: dict[str, str | None] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Validates the settings and normalizes paths and properties.

        Raises:
            TypeError: If a value or configuration property has an invalid
                type.
            ValueError: If a value is outside its accepted range, ``city`` is
                provided without ``state``, or configuration attempts to set
                the output directory or select an exporter.
            FileNotFoundError: If ``local_config_path`` cannot be read.
        """
        for name in ("population", "seed", "clinician_seed", "single_person_seed"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TypeError(f"{name} must be an int or None")
        if self.population is not None and self.population < 1:
            raise ValueError("population must be at least 1")
        for name in ("reference_date", "end_date", "age_range"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a str or None")
        if self.city and not self.state:
            raise ValueError("city requires state")
        if self.gender is not None and self.gender not in {"M", "F"}:
            raise ValueError("gender must be 'M', 'F', or None")
        if self.overflow_population is not None and not isinstance(
            self.overflow_population, bool
        ):
            raise TypeError("overflow_population must be a bool or None")
        if self.update_time_period is not None and (
            not isinstance(self.update_time_period, int)
            or isinstance(self.update_time_period, bool)
        ):
            raise TypeError("update_time_period must be an int or None")
        if self.update_time_period is not None and self.update_time_period < 1:
            raise ValueError("update_time_period must be at least 1")

        for name in (
            "local_config_path",
            "local_modules_dir",
            "initial_population_snapshot_path",
            "updated_population_snapshot_path",
            "fixed_record_path",
            "keep_matching_patients_path",
        ):
            object.__setattr__(self, name, _optional_path(getattr(self, name)))

        if not isinstance(self.synthea_config, Mapping):
            raise TypeError("synthea_config must be a mapping")
        config = {}
        for key, value in self.synthea_config.items():
            if not isinstance(key, str) or not _PROPERTY_KEY.fullmatch(key):
                raise TypeError(f"invalid Synthea property name: {key!r}")
            if isinstance(value, bool):
                config[key] = "true" if value else "false"
            elif isinstance(value, (str, int, float)):
                config[key] = str(value)
            else:
                raise TypeError(f"Synthea property {key!r} must have a scalar value")
        managed = config.keys() & _MANAGED_CONFIG_KEYS
        if managed:
            raise ValueError(
                "Synthea manages the output directory and exporter selection; "
                "remove: " + ", ".join(sorted(managed))
            )
        object.__setattr__(self, "synthea_config", config)

        if self.local_config_path is not None:
            try:
                text = self.local_config_path.read_text(encoding="utf-8")
            except OSError as error:
                raise FileNotFoundError(
                    f"Synthea config file not found: {self.local_config_path}"
                ) from error
            object.__setattr__(self, "local_config", _parse_properties(text))

    def fingerprint(self) -> str:
        """Hashes the settings into a stable output directory name.

        Returns:
            str: SHA-256 hex digest of the release version and all settings.
        """
        payload = json.dumps(
            {"version": _RELEASE.version, **asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class Synthea(BaseModel):
    """Generates Synthea populations in CSV format.

    The constructor only configures the generator: where output goes and how
    to find Java and the Synthea JAR. Nothing runs until :meth:`generate`,
    which takes the population settings, so one instance can produce many
    populations. Each call returns a directory of CSV files for
    :class:`~pyhealth.datasets.SyntheaCSVDataset`.

    Synthea is a rule-based simulator run as a Java subprocess, not a trained
    network. It subclasses :class:`~pyhealth.models.BaseModel` for API
    consistency but has no trainable parameters and no forward pass. Unlike the
    other generators, :meth:`generate` writes CSV files to disk and returns
    their directory rather than records.

    Note:
        Java 17 or newer must be installed; PyHealth does not install it.
        The executable is taken from ``java_path``, then ``JAVA_HOME``, then
        ``java`` on ``PATH``, and its version is checked before running.
        Only :meth:`generate` needs Java; construction and :meth:`build_argv`
        do not. The Synthea JAR itself is downloaded and cached on first run.
        See the Setup section of the API docs.

    Examples:
        >>> from pyhealth.datasets import SyntheaCSVDataset
        >>> from pyhealth.models import Synthea
        >>> synthea = Synthea("./synthea-output")  # nothing is generated yet
        >>> small = synthea.generate(population=10, seed=42)
        >>> large = synthea.generate(population=100, seed=7, state="Ohio")
        >>> dataset = SyntheaCSVDataset(small, tables=["patients"])
    """

    def __init__(
        self,
        output_dir: str | Path,
        java_path: str | Path | None = None,
        jar_path: str | Path | None = None,
        auto_download: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Initializes a Synthea generator.

        Args:
            output_dir (str or Path): Parent directory for generated
                populations. Each population is written to a subdirectory
                named by a hash of its settings.
            java_path (str or Path, optional): Java executable override.
            jar_path (str or Path, optional): Synthea JAR override.
            auto_download (bool): Whether to download the pinned JAR when absent.
            timeout (float, optional): Subprocess timeout in seconds for each
                :meth:`generate` call.
        """
        super().__init__(dataset=None)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.java_path = _optional_path(java_path)
        self.jar_path = _optional_path(jar_path)
        self.auto_download = auto_download
        self.timeout = timeout

    def forward(self, **kwargs) -> dict:
        """Rejects forward calls; Synthea has no forward pass.

        Args:
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always. Use :meth:`generate` to produce data.
        """
        raise NotImplementedError(
            "Synthea is a simulator with no forward pass; call generate() instead."
        )

    def extra_repr(self) -> str:
        """Summarizes the generator configuration shown by ``repr``.

        Returns:
            str: The output directory.
        """
        return f"output_dir={str(self.output_dir)!r}"

    def _resolve_java(self) -> Path:
        """Finds the Java executable used to run Synthea.

        The first executable found among ``java_path``, ``JAVA_HOME``, and
        ``PATH`` is used, and must be at least the pinned release's minimum
        Java version. An older Java is reported rather than skipped, so a stale
        ``JAVA_HOME`` is not silently bypassed.

        Returns:
            Path: The resolved Java executable.

        Raises:
            RuntimeError: If no Java executable can be found, it cannot be run,
                or its version is too old.
        """
        candidates = []
        if self.java_path:
            candidates.append(self.java_path)
        if java_home := os.environ.get("JAVA_HOME"):
            candidates.append(Path(java_home) / "bin" / "java")
        if java := shutil.which("java"):
            candidates.append(Path(java))
        minimum = _RELEASE.min_java_version
        for candidate in candidates:
            if candidate.is_file():
                java = candidate.resolve()
                version = self._java_major_version(java)
                if version is not None and version < minimum:
                    raise RuntimeError(
                        f"Synthea requires Java {minimum}+, but {java} is Java "
                        f"{version}. Install a JDK {minimum}+ (for example from "
                        "https://adoptium.net) or pass java_path=."
                    )
                return java
        raise RuntimeError(
            f"Synthea requires Java {minimum}+, but none was found via "
            "java_path, JAVA_HOME, or PATH. Install a JDK "
            f"{minimum}+ (for example from https://adoptium.net) or pass "
            "java_path=."
        )

    @staticmethod
    def _java_major_version(java: Path) -> int | None:
        """Reads the major version reported by ``java -version``.

        Handles both the modern scheme (``"17.0.2"`` -> 17) and the legacy one
        (``"1.8.0_392"`` -> 8).

        Args:
            java (Path): Java executable to query.

        Returns:
            int | None: The major version, or ``None`` if the output could not
            be parsed. Unparseable output is logged and tolerated rather than
            treated as a failure.

        Raises:
            RuntimeError: If the executable cannot be run.
        """
        try:
            result = subprocess.run(
                [str(java), "-version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"cannot run Java at {java}: {error}") from error
        match = _JAVA_VERSION.search(result.stderr + result.stdout)
        if not match:
            logger.warning("Could not determine the version of Java at %s", java)
            return None
        major, minor = match.groups()
        return int(minor) if major == "1" and minor else int(major)

    @classmethod
    def _resolve_jar_path(
        cls, jar_path: str | Path | None, auto_download: bool
    ) -> Path:
        """Resolves or downloads the pinned Synthea JAR.

        Args:
            jar_path (str or Path, optional): Explicit Synthea JAR path.
            auto_download (bool): Whether to download the pinned JAR when it is
                absent from the PyHealth cache.

        Returns:
            Path: The resolved Synthea JAR.

        Raises:
            FileNotFoundError: If an explicit JAR is missing or downloading is
                disabled and no cached JAR exists.
            RuntimeError: If a downloaded JAR fails checksum verification.
        """
        if jar_path:
            jar_path = Path(jar_path).expanduser()
            if not jar_path.is_file():
                raise FileNotFoundError(f"Synthea jar not found: {jar_path}")
            return jar_path.resolve()

        cache = Path(platformdirs.user_cache_dir("pyhealth")) / "synthea"
        jar = cache / f"synthea-with-dependencies-{_RELEASE.version}.jar"
        if jar.is_file():
            return jar
        if not auto_download:
            raise FileNotFoundError(
                f"Synthea jar not found at {jar}; pass jar_path= or enable download"
            )

        logger.warning("Downloading Synthea %s to %s", _RELEASE.version, jar)
        cache.mkdir(parents=True, exist_ok=True)
        partial = jar.with_suffix(".jar.part")
        try:
            urllib.request.urlretrieve(_RELEASE.url, partial)
            if cls._sha256(partial) != _RELEASE.sha256:
                raise RuntimeError("downloaded Synthea jar failed SHA256 verification")
            partial.replace(jar)
        finally:
            partial.unlink(missing_ok=True)
        return jar

    def _resolve_jar(self) -> Path:
        """Resolves the JAR using this instance's configured options.

        Returns:
            Path: The resolved Synthea JAR.

        Raises:
            FileNotFoundError: If the JAR is unavailable and cannot be downloaded.
            RuntimeError: If a downloaded JAR fails checksum verification.
        """
        return self._resolve_jar_path(self.jar_path, self.auto_download)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Computes the SHA-256 digest of a file.

        Args:
            path (Path): File to hash.

        Returns:
            str: Lowercase hexadecimal SHA-256 digest.
        """
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_jar_config(jar: Path) -> dict[str, str | None]:
        """Reads ``synthea.properties`` from a Synthea JAR.

        Args:
            jar (Path): Synthea JAR to inspect.

        Returns:
            dict[str, str | None]: Properties supported by the JAR.

        Raises:
            RuntimeError: If the JAR or its properties resource cannot be read.
        """
        try:
            with zipfile.ZipFile(jar) as archive:
                text = archive.read("synthea.properties").decode("utf-8")
        except (KeyError, OSError, zipfile.BadZipFile) as error:
            raise RuntimeError(f"cannot read synthea.properties from {jar}") from error
        return _parse_properties(text)

    @classmethod
    def get_available_config(
        cls,
        jar_path: str | Path | None = None,
        auto_download: bool = True,
        pattern: str = "*",
    ) -> dict[str, str | None]:
        """Returns properties supported by a Synthea JAR.

        Args:
            jar_path (str or Path, optional): Explicit Synthea JAR to inspect.
            auto_download (bool): Whether to download the pinned JAR when absent.
            pattern (str): Shell-style pattern used to filter property names.

        Returns:
            dict[str, str | None]: Matching properties and their defaults.
            Properties present only as comments have a value of ``None``.

        Raises:
            FileNotFoundError: If the JAR is unavailable and cannot be downloaded.
            RuntimeError: If the JAR or its properties resource cannot be read.
        """
        config = cls._read_jar_config(cls._resolve_jar_path(jar_path, auto_download))
        return {
            key: value for key, value in config.items() if fnmatchcase(key, pattern)
        }

    @classmethod
    def _validate_config_keys(cls, jar: Path, settings: _GenerationSettings) -> None:
        """Checks configured property names against the active JAR.

        Args:
            jar (Path): Synthea JAR whose properties define the accepted keys.
            settings (_GenerationSettings): Settings whose properties to check.

        Raises:
            ValueError: If a configured property is not supported by the JAR.
            RuntimeError: If the JAR properties cannot be read.
        """
        supported = cls._read_jar_config(jar)
        supplied = settings.synthea_config.keys() | settings.local_config.keys()
        unknown = supplied - supported.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"properties not supported by this Synthea jar: {names}")

    def _generation_dir(self, settings: _GenerationSettings) -> Path:
        """Returns the directory Synthea writes a population to.

        Args:
            settings (_GenerationSettings): Settings of the population.

        Returns:
            Path: ``output_dir`` joined with the settings fingerprint.
        """
        return self.output_dir / settings.fingerprint()

    @staticmethod
    def _find_csv_dir(generation_dir: Path) -> Path | None:
        """Finds the directory holding a population's CSV files.

        With Synthea's folder-per-run option enabled, the most recently
        modified directory containing ``patients.csv`` is selected.

        Args:
            generation_dir (Path): Directory Synthea wrote the population to.

        Returns:
            Path | None: The direct or nested CSV directory, or ``None`` if no
            ``patients.csv`` exists yet.
        """
        root = generation_dir / "csv"
        if (root / "patients.csv").is_file():
            return root
        candidates = list(root.rglob("patients.csv")) if root.is_dir() else []
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns).parent

    def _build_argv(
        self, java: Path, jar: Path, settings: _GenerationSettings
    ) -> list[str]:
        """Builds the Synthea subprocess argument vector for given settings.

        Args:
            java (Path): Java executable to invoke.
            jar (Path): Synthea JAR to execute.
            settings (_GenerationSettings): Settings of the population.

        Returns:
            list[str]: Complete subprocess argument vector.
        """
        config = dict(settings.synthea_config)
        config[_RELEASE.base_directory_key] = str(self._generation_dir(settings))
        config[_RELEASE.csv_export_key] = "true"
        for key in _RELEASE.unsupported_export_flags:
            config[key] = "false"

        argv = [str(java), "-jar", str(jar)]
        if settings.local_config_path is not None:
            argv += ["-c", str(settings.local_config_path)]
        argv += [f"--{key}={value}" for key, value in sorted(config.items())]
        options = (
            ("-s", settings.seed),
            ("-cs", settings.clinician_seed),
            ("-ps", settings.single_person_seed),
            ("-p", settings.population),
            ("-r", settings.reference_date),
            ("-e", settings.end_date),
            ("-g", settings.gender),
            ("-a", settings.age_range),
            ("-o", settings.overflow_population),
            ("-d", settings.local_modules_dir),
            ("-i", settings.initial_population_snapshot_path),
            ("-u", settings.updated_population_snapshot_path),
            ("-t", settings.update_time_period),
            ("-f", settings.fixed_record_path),
            ("-k", settings.keep_matching_patients_path),
        )
        for flag, value in options:
            if isinstance(value, bool):
                argv += [flag, "true" if value else "false"]
            elif value is not None:
                argv += [flag, str(value)]
        if settings.state:
            argv.append(settings.state)
        if settings.city:
            argv.append(settings.city)
        return argv

    def build_argv(self, java: Path, jar: Path, **settings) -> list[str]:
        """Builds the Synthea command line without running it.

        Useful for checking what :meth:`generate` would execute. Unlike
        :meth:`generate`, an omitted ``seed`` is left out of the command
        rather than chosen at random.

        Args:
            java (Path): Java executable to invoke.
            jar (Path): Synthea JAR to execute.
            **settings: Population settings, accepting the same keyword
                arguments as :meth:`generate` except ``overwrite``.

        Returns:
            list[str]: Complete subprocess argument vector.

        Raises:
            TypeError: If a setting is unknown or has an invalid type.
            ValueError: If a setting is outside its accepted range.
        """
        return self._build_argv(java, jar, _GenerationSettings(**settings))

    def generate(
        self,
        population: int | None = None,
        seed: int | None = None,
        state: str | None = None,
        city: str | None = None,
        clinician_seed: int | None = None,
        single_person_seed: int | None = None,
        reference_date: str | None = None,
        end_date: str | None = None,
        gender: str | None = None,
        age_range: str | None = None,
        overflow_population: bool | None = None,
        local_config_path: str | Path | None = None,
        local_modules_dir: str | Path | None = None,
        initial_population_snapshot_path: str | Path | None = None,
        updated_population_snapshot_path: str | Path | None = None,
        update_time_period: int | None = None,
        fixed_record_path: str | Path | None = None,
        keep_matching_patients_path: str | Path | None = None,
        synthea_config: Mapping[str, str | int | float | bool] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Generates a population and returns the directory of its CSV files.

        Each distinct combination of settings is written to its own
        subdirectory of ``output_dir``. If that subdirectory already holds
        output, it is returned without running Synthea again, unless
        ``overwrite`` is set. An omitted command-line option is not passed
        to Synthea, so Synthea's own default applies.

        Args:
            population (int, optional): ``-p`` population size.
            seed (int, optional): ``-s`` random seed. When omitted, a random
                seed is chosen and logged, so each call produces a new
                population. Pass a seed for reproducible output.
            state (str, optional): State positional argument.
            city (str, optional): City positional argument; requires ``state``.
            clinician_seed (int, optional): ``-cs`` clinician random seed.
            single_person_seed (int, optional): ``-ps`` single-person seed.
            reference_date (str, optional): ``-r`` date in YYYYMMDD form.
            end_date (str, optional): ``-e`` date in YYYYMMDD form.
            gender (str, optional): ``-g`` gender, either ``M`` or ``F``.
            age_range (str, optional): ``-a`` range in ``min-max`` form.
            overflow_population (bool, optional): ``-o`` population overflow
                switch.
            local_config_path (str or Path, optional): ``-c`` properties file.
            local_modules_dir (str or Path, optional): ``-d`` local module
                directory.
            initial_population_snapshot_path (str or Path, optional): ``-i``
                snapshot to load.
            updated_population_snapshot_path (str or Path, optional): ``-u``
                destination for the updated snapshot.
            update_time_period (int, optional): ``-t`` update period in days.
            fixed_record_path (str or Path, optional): ``-f`` fixed-demographics
                file.
            keep_matching_patients_path (str or Path, optional): ``-k`` keep
                module.
            synthea_config (Mapping, optional): ``synthea.properties``
                overrides. CSV export and the output directory are managed by
                this class and cannot be set here.
            overwrite (bool): Whether to run Synthea even if output for these
                settings already exists.

        Returns:
            Path: Directory containing the generated CSV files.

        Raises:
            TypeError: If a setting or configuration property has an invalid
                type.
            ValueError: If a setting is outside its accepted range, ``city`` is
                provided without ``state``, configuration sets the output
                directory or an exporter, or a property is unsupported by the
                active JAR.
            FileNotFoundError: If ``local_config_path`` cannot be read or the
                JAR is unavailable.
            RuntimeError: If Java cannot be resolved, Synthea exits
                unsuccessfully, or ``patients.csv`` is not produced.
            subprocess.TimeoutExpired: If generation exceeds ``timeout``.
        """
        chosen_seed = secrets.randbits(63) if seed is None else seed
        settings = _GenerationSettings(
            population=population,
            seed=chosen_seed,
            state=state,
            city=city,
            clinician_seed=clinician_seed,
            single_person_seed=single_person_seed,
            reference_date=reference_date,
            end_date=end_date,
            gender=gender,
            age_range=age_range,
            overflow_population=overflow_population,
            local_config_path=local_config_path,
            local_modules_dir=local_modules_dir,
            initial_population_snapshot_path=initial_population_snapshot_path,
            updated_population_snapshot_path=updated_population_snapshot_path,
            update_time_period=update_time_period,
            fixed_record_path=fixed_record_path,
            keep_matching_patients_path=keep_matching_patients_path,
            synthea_config=synthea_config or {},
        )
        if seed is None:
            logger.info("No seed given; using seed=%d", chosen_seed)

        generation_dir = self._generation_dir(settings)
        existing = self._find_csv_dir(generation_dir)
        if existing is not None and not overwrite:
            logger.info("Reusing existing Synthea output at %s", existing)
            return existing

        java = self._resolve_java()
        jar = self._resolve_jar()
        if settings.synthea_config or settings.local_config_path:
            self._validate_config_keys(jar, settings)
        generation_dir.mkdir(parents=True, exist_ok=True)
        argv = self._build_argv(java, jar, settings)
        logger.info("Running Synthea: %s", " ".join(argv))
        result = subprocess.run(argv, timeout=self.timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"Synthea exited with status {result.returncode}")

        csv_dir = self._find_csv_dir(generation_dir)
        if csv_dir is None:
            raise RuntimeError("Synthea did not generate CSV output")
        return csv_dir
