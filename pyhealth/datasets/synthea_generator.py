"""Generate Synthea populations in one or more supported output formats."""

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
_FHIR_EXPORT_KEY = str(_config_keys["fhir_export"])
_FHIR_BULK_DATA_KEY = str(_config_keys["fhir_bulk_data"])
UNSUPPORTED_EXPORT_FLAGS = tuple(
    str(flag) for flag in _release["unsupported_export_flags"]
)

_PROPERTY_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


class SyntheaGenerator:
    """Generates a Synthea population without imposing a dataset representation.

    Output selection uses Synthea's own configuration properties. Bulk FHIR R4
    is the default. ``exporter.csv.export=true`` adds CSV; setting
    ``exporter.fhir.export=false`` at the same time selects CSV only.
    """

    def __init__(
        self,
        output_dir: str | Path,
        population: int = 100,
        seed: int | None = None,
        state: str | None = None,
        city: str | None = None,
        java_path: str | Path | None = None,
        jar_path: str | Path | None = None,
        auto_download: bool = True,
        synthea_config: Mapping[str, str | int | float | bool] | None = None,
        timeout: float | None = None,
        regenerate: bool = False,
    ) -> None:
        if not isinstance(population, int) or isinstance(population, bool):
            raise TypeError("population must be an int")
        if population < 1:
            raise ValueError("population must be at least 1")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an int or None")
        if city and not state:
            raise ValueError("city requires state")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.population = population
        self.seed = seed
        self.state = state
        self.city = city
        self.java_path = Path(java_path).expanduser() if java_path else None
        self.jar_path = Path(jar_path).expanduser() if jar_path else None
        self.auto_download = auto_download
        self.synthea_config = self._normalize_config(synthea_config or {})
        self.csv_enabled = self._config_bool(
            self.synthea_config, _CSV_EXPORT_KEY, False
        )
        self.fhir_enabled = self._config_bool(
            self.synthea_config, _FHIR_EXPORT_KEY, True
        )
        self._validate_output_config()
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
                "csv": self.csv_enabled,
                "fhir": self.fhir_enabled,
                "population": self.population,
                "seed": self.seed,
                "state": self.state,
                "city": self.city,
                "config": fingerprint_config,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()

        configured_base = self.synthea_config.get(_BASE_DIRECTORY_KEY)
        self.generation_dir = (
            Path(configured_base).expanduser().resolve()
            if configured_base
            else self.output_dir / fingerprint
        )

    @staticmethod
    def _normalize_config(
        config: Mapping[str, str | int | float | bool],
    ) -> dict[str, str]:
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

    @staticmethod
    def _config_bool(config: Mapping[str, str], key: str, default: bool) -> bool:
        value = config.get(key)
        if value is None:
            return default
        normalized = value.lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Synthea output property {key!r} must be true or false")
        return normalized == "true"

    def _validate_output_config(self) -> None:
        blocked = [
            key
            for key in UNSUPPORTED_EXPORT_FLAGS
            if self._config_bool(self.synthea_config, key, False)
        ]
        if blocked:
            raise ValueError(
                "unsupported Synthea export flag(s): " + ", ".join(sorted(blocked))
            )

        bulk = self._config_bool(self.synthea_config, _FHIR_BULK_DATA_KEY, True)
        if self.fhir_enabled and not bulk:
            raise ValueError(f"SyntheaFHIRDataset requires {_FHIR_BULK_DATA_KEY}=true")
        if not self.csv_enabled and not self.fhir_enabled:
            raise ValueError("synthea_config must enable CSV, Bulk FHIR, or both")

    @property
    def outputs(self) -> dict[str, Path]:
        """Returns enabled output names and their output directories."""
        outputs = {}
        if self.csv_enabled:
            outputs["csv"] = self.resolved_output_path("csv")
        if self.fhir_enabled:
            outputs["fhir"] = self.resolved_output_path("fhir")
        return outputs

    def output_path(self, output: str) -> Path:
        """Returns the output directory for an enabled representation."""
        enabled = (output == "csv" and self.csv_enabled) or (
            output == "fhir" and self.fhir_enabled
        )
        if not enabled:
            raise ValueError(f"Synthea output {output!r} is not enabled")
        return self.generation_dir / output

    def resolved_output_path(self, output: str) -> Path:
        """Returns the directory containing an export, including nested CSV runs."""
        root = self.output_path(output)
        if output != "csv" or (root / "patients.csv").is_file():
            return root
        candidates = list(root.rglob("patients.csv")) if root.is_dir() else []
        if not candidates:
            return root
        latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        return latest.parent

    def _required_config(self) -> dict[str, str]:
        return {_FHIR_BULK_DATA_KEY: "true"} if self.fhir_enabled else {}

    def _effective_overrides(self) -> dict[str, str]:
        config = dict(self.synthea_config)
        config[_BASE_DIRECTORY_KEY] = str(self.generation_dir)
        config.update(self._required_config())
        return config

    def _resolve_java(self) -> Path:
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

    def _resolve_jar(self) -> Path:
        if self.jar_path:
            if not self.jar_path.is_file():
                raise FileNotFoundError(f"Synthea jar not found: {self.jar_path}")
            return self.jar_path.resolve()

        cache = Path(platformdirs.user_cache_dir("pyhealth")) / "synthea"
        jar = cache / f"synthea-with-dependencies-{SYNTHEA_VERSION}.jar"
        if jar.is_file():
            return jar
        if not self.auto_download:
            raise FileNotFoundError(
                f"Synthea jar not found at {jar}; pass jar_path= or enable download"
            )

        logger.warning("Downloading Synthea %s to %s", SYNTHEA_VERSION, jar)
        cache.mkdir(parents=True, exist_ok=True)
        partial = jar.with_suffix(".jar.part")
        try:
            urllib.request.urlretrieve(SYNTHEA_JAR_URL, partial)
            if self._sha256(partial) != SYNTHEA_JAR_SHA256:
                raise RuntimeError("downloaded Synthea jar failed SHA256 verification")
            partial.replace(jar)
        finally:
            partial.unlink(missing_ok=True)
        return jar

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_jar_config(jar: Path) -> dict[str, str | None]:
        try:
            with zipfile.ZipFile(jar) as archive:
                text = archive.read("synthea.properties").decode("utf-8")
        except (KeyError, OSError, zipfile.BadZipFile) as error:
            raise RuntimeError(f"cannot read synthea.properties from {jar}") from error

        config = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            disabled = line.startswith(("#", "!"))
            if disabled:
                line = line[1:].strip()
            key, separator, value = line.partition("=")
            if not separator:
                key, separator, value = line.partition(":")
            key = key.strip()
            if separator and _PROPERTY_KEY.fullmatch(key):
                config[key] = None if disabled else value.strip()
        return config

    def get_config(self, effective: bool = False) -> dict[str, str | None]:
        """Returns properties from the active jar, optionally with overrides."""
        config = self._read_jar_config(self._resolve_jar())
        unknown = self.synthea_config.keys() - config.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"properties not supported by this Synthea jar: {names}")
        if effective:
            config.update(self._effective_overrides())
        return config

    def show_config(self, pattern: str = "*") -> None:
        """Prints matching effective properties and where each value came from."""
        config = self.get_config(effective=True)
        generated = self._required_config()
        for key in sorted(key for key in config if fnmatchcase(key, pattern)):
            if key in generated:
                source = "output selection"
            elif key in self.synthea_config:
                source = "override"
            elif key == _BASE_DIRECTORY_KEY:
                source = "output_dir"
            elif config[key] is None:
                source = "available"
            else:
                source = "default"
            value = "<unset>" if config[key] is None else config[key]
            print(f"{key} = {value} [{source}]")

    def build_argv(self, java: Path, jar: Path) -> list[str]:
        """Builds the Synthea subprocess argument vector."""
        config = self._effective_overrides()
        argv = [str(java), "-jar", str(jar)]
        argv += [f"--{key}={value}" for key, value in sorted(config.items())]
        argv += ["-p", str(self.population)]
        if self.seed is not None:
            argv += ["-s", str(self.seed)]
        if self.state:
            argv.append(self.state)
        if self.city:
            argv.append(self.city)
        return argv

    def _has_output(self, output: str) -> bool:
        root = self.resolved_output_path(output)
        if output == "csv":
            return (root / "patients.csv").is_file()
        return root.is_dir() and next(root.rglob("*.ndjson"), None) is not None

    def ensure_generated(self) -> dict[str, Path]:
        """Generates once when requested output is absent, then returns paths."""
        missing = any(not self._has_output(output) for output in self.outputs)
        if not self._generated and (self.regenerate or missing):
            self.run()
            self._generated = True
        return self.outputs

    def run(self) -> dict[str, Path]:
        """Runs Synthea and returns the requested output directories."""
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        java = self._resolve_java()
        jar = self._resolve_jar()
        if self.synthea_config:
            self.get_config()
        argv = self.build_argv(java, jar)
        logger.info("Running Synthea: %s", " ".join(argv))
        result = subprocess.run(argv, timeout=self.timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"Synthea exited with status {result.returncode}")

        missing = [
            output for output in sorted(self.outputs) if not self._has_output(output)
        ]
        if missing:
            raise RuntimeError(
                "Synthea did not generate requested output(s): " + ", ".join(missing)
            )
        return self.outputs
