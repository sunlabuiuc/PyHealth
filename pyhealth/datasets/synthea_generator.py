"""Generate Synthea populations as CSV files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType

import platformdirs
import yaml

logger = logging.getLogger(__name__)

_RELEASE_CONFIG_PATH = Path(__file__).parent / "configs" / "synthea_release.yaml"
with _RELEASE_CONFIG_PATH.open(encoding="utf-8") as _release_file:
    _release = yaml.safe_load(_release_file)

SYNTHEA_VERSION = str(_release["version"])
SYNTHEA_JAR_URL = str(_release["url"])
SYNTHEA_JAR_SHA256 = str(_release["sha256"])
if not re.fullmatch(r"[0-9a-f]{64}", SYNTHEA_JAR_SHA256):
    raise RuntimeError(f"invalid Synthea SHA-256 in {_RELEASE_CONFIG_PATH}")

_config_keys = _release["config_keys"]
_BASE_DIRECTORY_KEY = str(_config_keys["base_directory"])
_CSV_EXPORT_KEY = str(_config_keys["csv_export"])
UNSUPPORTED_EXPORT_FLAGS = tuple(
    str(flag) for flag in _release["unsupported_export_flags"]
)

_PROPERTY_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


class SyntheaGenerator:
    """Generates a Synthea population in CSV format.

    Synthea command-line options are exposed as constructor parameters. Settings
    from ``synthea.properties`` belong in ``synthea_config``. CSV output is
    enabled automatically. An omitted command-line option is not emitted, so
    Synthea supplies its native default.
    """

    def __init__(
        self,
        output_dir: str | Path,
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
        java_path: str | Path | None = None,
        jar_path: str | Path | None = None,
        auto_download: bool = True,
        synthea_config: Mapping[str, str | int | float | bool] | None = None,
        timeout: float | None = None,
        regenerate: bool = False,
    ) -> None:
        """Initializes a CSV population generator.

        Args:
            output_dir (str or Path): Parent directory for fingerprinted
                generations.
            population (int, optional): ``-p`` population size.
            seed (int, optional): ``-s`` random seed.
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
            java_path (str or Path, optional): Java executable override.
            jar_path (str or Path, optional): Synthea JAR override.
            auto_download (bool): Whether to download the pinned JAR when absent.
            synthea_config (Mapping, optional): ``synthea.properties`` overrides.
            timeout (float, optional): Subprocess timeout in seconds.
            regenerate (bool): Whether the first lazy access replaces existing
                output.

        Raises:
            TypeError: If a constructor value or configuration property has an
                invalid type.
            ValueError: If a value is outside its accepted range, ``city`` is
                provided without ``state``, or configuration attempts to select
                an exporter.
            FileNotFoundError: If ``local_config_path`` cannot be read.
        """
        if population is not None and (
            not isinstance(population, int) or isinstance(population, bool)
        ):
            raise TypeError("population must be an int or None")
        if population is not None and population < 1:
            raise ValueError("population must be at least 1")
        for name, value in (
            ("seed", seed),
            ("clinician_seed", clinician_seed),
            ("single_person_seed", single_person_seed),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TypeError(f"{name} must be an int or None")
        for name, value in (
            ("reference_date", reference_date),
            ("end_date", end_date),
            ("age_range", age_range),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a str or None")
        if city and not state:
            raise ValueError("city requires state")
        if gender is not None and gender not in {"M", "F"}:
            raise ValueError("gender must be 'M', 'F', or None")
        if overflow_population is not None and not isinstance(
            overflow_population, bool
        ):
            raise TypeError("overflow_population must be a bool or None")
        if update_time_period is not None and (
            not isinstance(update_time_period, int)
            or isinstance(update_time_period, bool)
        ):
            raise TypeError("update_time_period must be an int or None")
        if update_time_period is not None and update_time_period < 1:
            raise ValueError("update_time_period must be at least 1")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.population = population
        self.seed = seed
        self.state = state
        self.city = city
        self.clinician_seed = clinician_seed
        self.single_person_seed = single_person_seed
        self.reference_date = reference_date
        self.end_date = end_date
        self.gender = gender
        self.age_range = age_range
        self.overflow_population = overflow_population
        self.local_config_path = self._optional_path(local_config_path)
        self.local_modules_dir = self._optional_path(local_modules_dir)
        self.initial_population_snapshot_path = self._optional_path(
            initial_population_snapshot_path
        )
        self.updated_population_snapshot_path = self._optional_path(
            updated_population_snapshot_path
        )
        self.update_time_period = update_time_period
        self.fixed_record_path = self._optional_path(fixed_record_path)
        self.keep_matching_patients_path = self._optional_path(
            keep_matching_patients_path
        )
        self.java_path = Path(java_path).expanduser() if java_path else None
        self.jar_path = Path(jar_path).expanduser() if jar_path else None
        self.auto_download = auto_download
        normalized_config = self._normalize_config(synthea_config or {})
        managed_export_keys = normalized_config.keys() & {
            _CSV_EXPORT_KEY,
            *UNSUPPORTED_EXPORT_FLAGS,
        }
        if managed_export_keys:
            names = ", ".join(sorted(managed_export_keys))
            raise ValueError(
                "SyntheaGenerator manages CSV exporter selection; remove: " + names
            )
        self.synthea_config = MappingProxyType(normalized_config)
        local_config = (
            self._read_config_file(self.local_config_path)
            if self.local_config_path
            else {}
        )
        self._local_config = MappingProxyType(local_config)
        effective_config = dict(local_config)
        effective_config.update(normalized_config)
        self.timeout = timeout
        self.regenerate = regenerate
        self._generated = False

        fingerprint_config = {
            key: value
            for key, value in self.synthea_config.items()
            if key != _BASE_DIRECTORY_KEY
        }
        fingerprint_payload = json.dumps(
            {
                "version": SYNTHEA_VERSION,
                "population": self.population,
                "seed": self.seed,
                "clinician_seed": self.clinician_seed,
                "single_person_seed": self.single_person_seed,
                "reference_date": self.reference_date,
                "end_date": self.end_date,
                "gender": self.gender,
                "age_range": self.age_range,
                "overflow_population": self.overflow_population,
                "local_config_path": self._path_string(self.local_config_path),
                "local_modules_dir": self._path_string(self.local_modules_dir),
                "initial_population_snapshot_path": self._path_string(
                    self.initial_population_snapshot_path
                ),
                "updated_population_snapshot_path": self._path_string(
                    self.updated_population_snapshot_path
                ),
                "update_time_period": self.update_time_period,
                "fixed_record_path": self._path_string(self.fixed_record_path),
                "keep_matching_patients_path": self._path_string(
                    self.keep_matching_patients_path
                ),
                "state": self.state,
                "city": self.city,
                "local_config": local_config,
                "config": fingerprint_config,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()

        configured_base = effective_config.get(_BASE_DIRECTORY_KEY)
        self.generation_dir = (
            Path(configured_base).expanduser().resolve()
            if configured_base
            else self.output_dir / fingerprint
        )

    @staticmethod
    def _optional_path(value: str | Path | None) -> Path | None:
        """Converts an optional path-like value to an expanded path.

        Args:
            value (str or Path, optional): Path-like value to convert.

        Returns:
            Path | None: The expanded path, or ``None`` when omitted.
        """
        return Path(value).expanduser() if value is not None else None

    @staticmethod
    def _path_string(value: Path | None) -> str | None:
        """Converts an optional path to its string representation.

        Args:
            value (Path, optional): Path to convert.

        Returns:
            str | None: The path string, or ``None`` when omitted.
        """
        return str(value) if value is not None else None

    @staticmethod
    def _normalize_config(
        config: Mapping[str, str | int | float | bool],
    ) -> dict[str, str]:
        """Validates and serializes Synthea property overrides.

        Args:
            config (Mapping): Property names mapped to scalar Python values.

        Returns:
            dict[str, str]: Configuration values serialized for Synthea's CLI.

        Raises:
            TypeError: If a property name is invalid or a value is not a
                supported scalar.
        """
        normalized = {}
        for key, value in config.items():
            if not isinstance(key, str) or not _PROPERTY_KEY.fullmatch(key):
                raise TypeError(f"invalid Synthea property name: {key!r}")
            if isinstance(value, bool):
                normalized[key] = "true" if value else "false"
            elif isinstance(value, (str, int, float)):
                normalized[key] = str(value)
            else:
                raise TypeError(f"Synthea property {key!r} must have a scalar value")
        return normalized

    def output_path(self) -> Path:
        """Returns the configured CSV output directory.

        Returns:
            Path: The CSV directory beneath the fingerprinted generation path.
        """
        return self.generation_dir / "csv"

    def resolved_output_path(self) -> Path:
        """Resolves the directory containing generated CSV files.

        When Synthea's folder-per-run option is enabled, this method selects the
        most recently modified directory containing ``patients.csv``.

        Returns:
            Path: The direct or nested CSV output directory.
        """
        root = self.output_path()
        if (root / "patients.csv").is_file():
            return root
        candidates = list(root.rglob("patients.csv")) if root.is_dir() else []
        if not candidates:
            return root
        latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        return latest.parent

    def _effective_overrides(self) -> dict[str, str]:
        """Builds the property overrides required for a CSV-only run.

        Returns:
            dict[str, str]: User properties plus the output directory and
            wrapper-managed exporter settings.
        """
        config = dict(self.synthea_config)
        config[_BASE_DIRECTORY_KEY] = str(self.generation_dir)
        config[_CSV_EXPORT_KEY] = "true"
        for key in UNSUPPORTED_EXPORT_FLAGS:
            config[key] = "false"
        return config

    def with_config(
        self,
        config: Mapping[str, str | int | float | bool],
    ) -> SyntheaGenerator:
        """Returns a new generator with merged Synthea property overrides.

        The current generator is not modified. The new generator repeats normal
        constructor validation and receives a freshly computed output fingerprint.

        Args:
            config (Mapping): Properties to add or replace.

        Returns:
            SyntheaGenerator: A new generator containing the merged properties.

        Raises:
            TypeError: If ``config`` is not a mapping or contains invalid values.
            ValueError: If ``config`` attempts to select an exporter.
        """
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        merged = dict(self.synthea_config)
        merged.update(config)
        return type(self)(
            output_dir=self.output_dir,
            population=self.population,
            seed=self.seed,
            state=self.state,
            city=self.city,
            clinician_seed=self.clinician_seed,
            single_person_seed=self.single_person_seed,
            reference_date=self.reference_date,
            end_date=self.end_date,
            gender=self.gender,
            age_range=self.age_range,
            overflow_population=self.overflow_population,
            local_config_path=self.local_config_path,
            local_modules_dir=self.local_modules_dir,
            initial_population_snapshot_path=self.initial_population_snapshot_path,
            updated_population_snapshot_path=self.updated_population_snapshot_path,
            update_time_period=self.update_time_period,
            fixed_record_path=self.fixed_record_path,
            keep_matching_patients_path=self.keep_matching_patients_path,
            java_path=self.java_path,
            jar_path=self.jar_path,
            auto_download=self.auto_download,
            synthea_config=merged,
            timeout=self.timeout,
            regenerate=self.regenerate,
        )

    def _resolve_java(self) -> Path:
        """Finds the Java executable used to run Synthea.

        Resolution checks ``java_path``, ``JAVA_HOME``, and then the executable
        available on ``PATH``.

        Returns:
            Path: The resolved Java executable.

        Raises:
            RuntimeError: If no Java executable can be found.
        """
        candidates = []
        if self.java_path:
            candidates.append(self.java_path)
        if java_home := os.environ.get("JAVA_HOME"):
            candidates.append(Path(java_home) / "bin" / "java")
        if java := shutil.which("java"):
            candidates.append(Path(java))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise RuntimeError(
            "Synthea requires Java 17+. Install Java or pass java_path=."
        )

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
        jar = cache / f"synthea-with-dependencies-{SYNTHEA_VERSION}.jar"
        if jar.is_file():
            return jar
        if not auto_download:
            raise FileNotFoundError(
                f"Synthea jar not found at {jar}; pass jar_path= or enable download"
            )

        logger.warning("Downloading Synthea %s to %s", SYNTHEA_VERSION, jar)
        cache.mkdir(parents=True, exist_ok=True)
        partial = jar.with_suffix(".jar.part")
        try:
            urllib.request.urlretrieve(SYNTHEA_JAR_URL, partial)
            if cls._sha256(partial) != SYNTHEA_JAR_SHA256:
                raise RuntimeError("downloaded Synthea jar failed SHA256 verification")
            partial.replace(jar)
        finally:
            partial.unlink(missing_ok=True)
        return jar

    def _resolve_jar(self) -> Path:
        """Resolves the JAR using this generator's configured options.

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
    def _parse_config(text: str) -> dict[str, str | None]:
        """Parses Java properties text into a Python mapping.

        Commented properties are retained with a value of ``None`` so callers
        can discover properties that Synthea supports but leaves unset.

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

    @classmethod
    def _read_jar_config(cls, jar: Path) -> dict[str, str | None]:
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
        return cls._parse_config(text)

    @classmethod
    def _read_config_file(cls, path: Path) -> dict[str, str | None]:
        """Reads a local Synthea properties file.

        Args:
            path (Path): Properties file to read.

        Returns:
            dict[str, str | None]: Parsed local properties.

        Raises:
            FileNotFoundError: If the properties file cannot be read.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise FileNotFoundError(f"Synthea config file not found: {path}") from error
        return cls._parse_config(text)

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

    def _validate_config_keys(self, jar: Path) -> None:
        """Checks configured property names against the active JAR.

        Args:
            jar (Path): Synthea JAR whose properties define the accepted keys.

        Raises:
            ValueError: If a configured property is not supported by the JAR.
            RuntimeError: If the JAR properties cannot be read.
        """
        config = self._read_jar_config(jar)
        supplied_keys = self.synthea_config.keys() | self._local_config.keys()
        unknown = supplied_keys - config.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"properties not supported by this Synthea jar: {names}")

    def build_argv(self, java: Path, jar: Path) -> list[str]:
        """Builds the Synthea subprocess argument vector.

        Args:
            java (Path): Java executable to invoke.
            jar (Path): Synthea JAR to execute.

        Returns:
            list[str]: Complete subprocess argument vector.
        """
        config = self._effective_overrides()
        argv = [str(java), "-jar", str(jar)]
        if self.local_config_path is not None:
            argv += ["-c", str(self.local_config_path)]
        argv += [f"--{key}={value}" for key, value in sorted(config.items())]
        if self.seed is not None:
            argv += ["-s", str(self.seed)]
        if self.clinician_seed is not None:
            argv += ["-cs", str(self.clinician_seed)]
        if self.single_person_seed is not None:
            argv += ["-ps", str(self.single_person_seed)]
        if self.population is not None:
            argv += ["-p", str(self.population)]
        if self.reference_date is not None:
            argv += ["-r", self.reference_date]
        if self.end_date is not None:
            argv += ["-e", self.end_date]
        if self.gender is not None:
            argv += ["-g", self.gender]
        if self.age_range is not None:
            argv += ["-a", self.age_range]
        if self.overflow_population is not None:
            overflow = "true" if self.overflow_population else "false"
            argv += ["-o", overflow]
        if self.local_modules_dir is not None:
            argv += ["-d", str(self.local_modules_dir)]
        if self.initial_population_snapshot_path is not None:
            argv += ["-i", str(self.initial_population_snapshot_path)]
        if self.updated_population_snapshot_path is not None:
            argv += ["-u", str(self.updated_population_snapshot_path)]
        if self.update_time_period is not None:
            argv += ["-t", str(self.update_time_period)]
        if self.fixed_record_path is not None:
            argv += ["-f", str(self.fixed_record_path)]
        if self.keep_matching_patients_path is not None:
            argv += ["-k", str(self.keep_matching_patients_path)]
        if self.state:
            argv.append(self.state)
        if self.city:
            argv.append(self.city)
        return argv

    def _has_output(self) -> bool:
        """Checks whether the generated patient CSV exists.

        Returns:
            bool: ``True`` when ``patients.csv`` is available.
        """
        return (self.resolved_output_path() / "patients.csv").is_file()

    def ensure_generated(self) -> Path:
        """Ensures that CSV output exists and returns its directory.

        Existing output is reused unless ``regenerate`` requests replacement on
        the first access through this generator.

        Returns:
            Path: Directory containing the generated CSV files.

        Raises:
            RuntimeError: If Java is unavailable, Synthea fails, or expected CSV
                output is not produced.
        """
        if not self._generated and (self.regenerate or not self._has_output()):
            self.run()
            self._generated = True
        return self.resolved_output_path()

    def run(self) -> Path:
        """Runs Synthea and returns the generated CSV directory.

        Returns:
            Path: Directory containing the generated CSV files.

        Raises:
            FileNotFoundError: If the configured JAR is unavailable.
            ValueError: If a configured property is unsupported by the active JAR.
            RuntimeError: If Java cannot be resolved, Synthea exits unsuccessfully,
                or ``patients.csv`` is not produced.
            subprocess.TimeoutExpired: If generation exceeds ``timeout``.
        """
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        java = self._resolve_java()
        jar = self._resolve_jar()
        if self.synthea_config or self.local_config_path:
            self._validate_config_keys(jar)
        argv = self.build_argv(java, jar)
        logger.info("Running Synthea: %s", " ".join(argv))
        result = subprocess.run(argv, timeout=self.timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"Synthea exited with status {result.returncode}")

        if not self._has_output():
            raise RuntimeError("Synthea did not generate CSV output")
        return self.resolved_output_path()
