import os
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

class LIDCIDRIDataset:
    """
    Download LIDC‑IDRI directly via TCIA REST API instead of NBIA CLI.

    Workflow
    --------
    1. Call `getSeries` to obtain **all SeriesInstanceUIDs** for the LIDC‑IDRI collection.
    2. Ask the user to confirm the ~133 GB transfer.
    3. For each UID, call `getImage` (TCIA REST) which returns a **zipped DICOM series**.
    4. Stream the zip, extract into `<root>/<SeriesUID>/`, and delete the zip.
       Skips any UID whose folder already exists.

    Notes
    -----
    * No external tools, manifests, or 414‑prone URLs — pure HTTPS downloads.
    * Parallel downloads (default 4 threads) for throughput; tweak `max_workers`.
    * Set `dev=True` to grab only the first 5 series for a quick smoke‑test.
    """

    SERIES_API = (
        "https://services.cancerimagingarchive.net/services/v4/TCIA/query/getSeries"
    )
    IMAGE_API = (
        "https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage"
    )

    def __init__(self, root: str = "./LIDC-IDRI", dev: bool = False, max_workers: int = 4):
        self.root = root
        self.dev = dev
        self.max_workers = max_workers
        os.makedirs(self.root, exist_ok=True)

        self.series = self._fetch_series()
        if self.dev:
            self.series = self.series[:5]
        self._confirm()
        self._download_all()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _fetch_series(self):
        print("Retrieving series list …")
        resp = requests.get(self.SERIES_API, params={"Collection": "LIDC-IDRI", "format": "json"})
        resp.raise_for_status()
        uids = [item["SeriesInstanceUID"] for item in resp.json()]
        print(f"Found {len(uids)} series in LIDC-IDRI")
        return uids

    def _confirm(self):
        size_hint = "~133 GB" if not self.dev else "a few MB (dev mode)"
        msg = (
            f"Destination : {self.root}\n"
            f"Series count: {len(self.series)}\n"
            f"Estimated size: {size_hint}\n"
            "Proceed with download? [y/N]: "
        )
        if input(msg).strip().lower() != "y":
            print("Download cancelled.")
            exit()

    def _download_one(self, uid):
        out_dir = os.path.join(self.root, uid)
        if os.path.isdir(out_dir):
            return f"Skipped (exists) {uid}"
        tmp_zip = os.path.join(self.root, f"{uid}.zip")
        try:
            with requests.get(self.IMAGE_API, params={"SeriesInstanceUID": uid}, stream=True) as r:
                r.raise_for_status()
                with open(tmp_zip, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            with zipfile.ZipFile(tmp_zip) as z:
                z.extractall(out_dir)
            return f"Downloaded {uid}"
        finally:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)

    def _download_all(self):
        print("Starting downloads …")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._download_one, uid): uid for uid in self.series}
            for future in as_completed(futures):
                print(future.result())
        print("All requested series processed.")
