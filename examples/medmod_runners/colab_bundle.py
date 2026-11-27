"""Zip the code and data that the Colab runs need, for your own Google Drive.

    cd "<repo>"
    PYTHONPATH="$PWD" python -m examples.medmod_runners.colab_bundle \\
        --data-root /path/to/mimic_data [--code-only] [--ihm]

Writes <data-root>/colab_upload/:
* convirt_data.zip: a mini data root with the same layout as the full one, only
  the files ``pretrain_samples.load_raw`` reads for the kept 7,755 / 882 stays:
  medmod_extracted/phenotyping/{train,val}_listfile.csv, the episode files of
  those stays (phenotyping/train/...), medmod_extracted/root/all_stays.csv,
  cohorts/pretrain_cohort.csv, and their resized images.
* convirt_code.zip: the code as it is in this folder now, tracked and new
  files that git does not ignore (pyhealth/, pyproject.toml, README.rst for
  ``pip install -e``, and the two MedMod examples), this folder's run scripts
  (convirt_pretrain, convirt_probe, early_fusion_run) and their helpers
  (pretrain_samples, samples, provenance), and code/CODE_VERSION.json with the
  time the zip was made. The runners record that time as the code version in
  every report and checkpoint. No data and no caches.

* ihm_data.zip (--ihm): what Early Fusion and the linear probe read, in the
  same layout: mimiciv_medmod_slim/1.0/ (the 7 slim tables),
  mimic-cxr-jpg/2.0.0/ (metadata, PyHealth metadata, CheXpert CSVs, all
  resized images), and medfuse_weights/{uni_ehr_all,cxr_rad_full}/ (MedFuse's
  released checkpoints, from --medfuse-weights).
* mimic4_medmod_ihm_colab_runs.ipynb: copied from examples/.

Upload the MIMIC data only to your own Google Drive, and do not share it with
anyone who lacks PhysioNet access. This script prints only counts and sizes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = ("__init__.py", "convirt_pretrain.py", "pretrain_samples.py",
         "provenance.py", "samples.py", "early_fusion_run.py", "convirt_probe.py")


def add(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    store = path.suffix.lower() in (".jpg", ".jpeg", ".png")
    zf.write(path, arcname, zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--code-only", action="store_true", help="skip convirt_data.zip")
    ap.add_argument("--ihm", action="store_true", help="also write ihm_data.zip")
    ap.add_argument(
        "--medfuse-weights",
        help="folder with MedFuse's uni_ehr_all/ and cxr_rad_full/ (needed with --ihm)",
    )
    args = ap.parse_args()
    root = Path(args.data_root)
    out = root / "colab_upload"
    out.mkdir(exist_ok=True)
    write_code(out)
    import shutil
    notebook = "mimic4_medmod_ihm_colab_runs.ipynb"
    shutil.copy2(REPO / "examples" / notebook, out / notebook)
    if args.ihm:
        if not args.medfuse_weights:
            ap.error("--ihm needs --medfuse-weights")
        write_ihm(root, Path(args.medfuse_weights), out)
    if not args.code_only:
        write_data(root, out)


def write_ihm(root: Path, weights: Path, out: Path) -> None:
    cxr = Path("mimic-cxr-jpg/2.0.0")
    files = sorted((root / "mimiciv_medmod_slim/1.0").rglob("*.csv.gz"))
    files += [root / cxr / n for n in ("mimic-cxr-2.0.0-metadata.csv.gz",
                                       "mimic-cxr-2.0.0-metadata-pyhealth.csv",
                                       "mimic-cxr-2.0.0-chexpert.csv.gz")]
    images = sorted(p for p in (root / cxr / "resized").glob("*.jpg")
                    if not p.name.startswith("."))
    zpath = out / "ihm_data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in files + images:
            add(zf, f, str(f.relative_to(root)))
        for name in ("uni_ehr_all", "cxr_rad_full"):
            f = weights / name / "best_checkpoint.pth.tar"
            add(zf, f, f"medfuse_weights/{name}/best_checkpoint.pth.tar")
    print(f"ihm_data.zip: {len(files)} tables, {len(images)} images, 2 MedFuse "
          f"checkpoints, {zpath.stat().st_size / 1e6:.0f} MB")


def write_code(out: Path) -> None:
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others",
         "--exclude-standard", "--", "pyhealth", "pyproject.toml", "README.rst",
         "examples/mimic4_medmod_ihm_early_fusion.py",
         "examples/mimic4_medmod_convirt.py"],
        capture_output=True, check=True).stdout.decode().split("\0")
    files = sorted({f for f in listed if f and (REPO / f).is_file()})
    files += [f"examples/medmod_runners/{n}" for n in TOOLS]
    version = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    code_zip = out / "convirt_code.zip"
    with zipfile.ZipFile(code_zip, "w") as zf:
        for f in files:
            add(zf, REPO / f, f"code/{f}")
        zf.writestr("code/CODE_VERSION.json", json.dumps(version, indent=1))
    print(f"convirt_code.zip: made {version['created']}, "
          f"{len(files) + 1} files, {code_zip.stat().st_size / 1e6:.1f} MB")


def write_data(root: Path, out: Path) -> None:
    from .pretrain_samples import VERIFIED, load_raw

    raw, _ = load_raw(root)
    counts = {s: sum(r["split"] == s for r in raw) for s in VERIFIED}
    if counts != VERIFIED:
        raise SystemExit(f"counts {counts} differ from {VERIFIED}")

    pheno = Path("medmod_extracted/phenotyping")
    files = {
        pheno / "train_listfile.csv",
        pheno / "val_listfile.csv",
        Path("medmod_extracted/root/all_stays.csv"),
        Path("cohorts/pretrain_cohort.csv"),
    }
    for r in raw:
        files.add(Path(r["_episode"]).relative_to(root))
        files.add(Path(r["image_path"]).relative_to(root))
    data_zip = out / "convirt_data.zip"
    with zipfile.ZipFile(data_zip, "w") as zf:
        for rel in sorted(files):
            add(zf, root / rel, str(rel))
    n_img = sum(1 for f in files if f.suffix == ".jpg")
    print(f"convirt_data.zip: {len(files)} files ({n_img} images, "
          f"{len(files) - n_img - 4} episode files, 4 tables), "
          f"{data_zip.stat().st_size / 1e6:.0f} MB; stays {counts}")


if __name__ == "__main__":
    main()
