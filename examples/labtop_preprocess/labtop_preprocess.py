import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer


class LabTOPPreprocessor:
    """
    Preprocesses MIMIC-IV data into LabTOP-style token sequences.

    This class:
      1. Loads ICU stays (cohort) from icustays.csv.gz.
      2. Loads and filters event tables (labevents, inputevents, outputevents,
         procedureevents, EMAR + EMAR_DETAIL).
      3. Joins patient demographics (gender, age, race).
      4. Restricts events to the first 72 hours after ICU admission.
      5. Converts each stay into a token sequence using a clinical BERT tokenizer,
         with additional special tokens for time, event types, and [EOE].
      6. Splits stays by subject_id into train / val / test and saves:
           - {train,val,test}.pkl
           - {val,test}_eval.pkl (prompt/label pairs for lab events)
           - tokenizer in out_dir/

    Args:
        data_dir: Path to MIMIC-IV data directory (with icustays.csv.gz, labevents.csv.gz, etc.).
        tokenizer_name: HuggingFace tokenizer name (default: "emilyalsentzer/Bio_ClinicalBERT").
        max_len: Maximum sequence length for tokenized stays.
        stay_limit: Optional limit on number of ICU stays (for quick demo / debugging).
        shard_size: Number of sequences per temporary shard before merging.
        inspect: If True, prints a few example events as raw text for sanity check.
        inspect_max_stays: Number of stays to print in inspect mode.
        inspect_max_events: Number of events per stay to print in inspect mode.
        out_dir: Directory to save tokenizer and processed .pkl files.
    """

    def __init__(
        self,
        data_dir,
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        max_len=1024,
        stay_limit=None,
        shard_size=50000,
        inspect=False,
        inspect_max_stays=3,
        inspect_max_events=5,
        out_dir="processed_data",
    ):
        self.data_dir = data_dir
        self.max_len = max_len
        self.stay_limit = stay_limit
        self.shard_size = shard_size
        self.inspect = inspect
        self.inspect_max_stays = inspect_max_stays
        self.inspect_max_events = inspect_max_events
        self.out_dir = out_dir

        # Use Bio-Clinical BERT tokenizer instead of GPT-2
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Make sure we have a pad token (Bio-ClinicalBERT already has [PAD], but keep this for safety)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token

        # Time tokens
        self.days = [f"[DAY{i}]" for i in range(4)]  # 0-3
        self.weekdays = ["[MON]", "[TUE]", "[WED]", "[THU]", "[FRI]", "[SAT]", "[SUN]"]
        self.hours = [f"[{i:02d}h]" for i in range(24)]
        # minute bins in 10-min steps: 0,10,20,30,40,50
        self.minutes = [f"[{i:02d}m]" for i in range(0, 60, 10)]

        # Event type tokens (plain tokens added to vocab)
        self.event_types = [
            "labevent",
            "inputevent",
            "outputevent",
            "procedureevent",
            "emarevent",
        ]

        # Other special tokens
        self.other_specials = ["[EOE]"]

        new_tokens = (
            self.days
            + self.weekdays
            + self.hours
            + self.minutes
            + self.event_types
            + self.other_specials
        )
        self.tokenizer.add_tokens(new_tokens)

        # Save tokenizer to out_dir so training/eval can load it
        os.makedirs(self.out_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.out_dir)

        # Cache for time tokens (day, weekday, hour, minute_bin) -> token string
        self.time_cache = {}

    # ---------------------------------------------------------------------
    # Basic loading functions
    # ---------------------------------------------------------------------
    def load_icustays(self):
        print("Loading ICU stays...")
        icu_path = os.path.join(self.data_dir, "icustays.csv.gz")
        df = pd.read_csv(
            icu_path,
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        )
        df["intime"] = pd.to_datetime(df["intime"])
        df["outtime"] = pd.to_datetime(df["outtime"])
        return df

    def load_events_chunked(
        self, filename, cols, date_col, hadm_ids_set, item_filter=None
    ):
        """
        Generic large-table loading helper:
          - Read only selected columns (cols).
          - Filter rows to hadm_id in hadm_ids_set.
          - Convert date_col to datetime.
          - Optionally keep only itemid in item_filter.
        """
        path = os.path.join(self.data_dir, filename)
        print(f"Reading {filename} from {path}...")
        chunksize = 1_000_000
        chunks = []

        for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
            # Filter by hadm_id
            if "hadm_id" in chunk.columns:
                chunk = chunk[chunk["hadm_id"].isin(hadm_ids_set)]
            if chunk.empty:
                continue

            chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
            chunk = chunk.dropna(subset=[date_col])
            if chunk.empty:
                continue

            # Optional item filter (for labs / procedures)
            if item_filter is not None and "itemid" in chunk.columns:
                chunk = chunk[chunk["itemid"].isin(item_filter)]
                if chunk.empty:
                    continue

            chunks.append(chunk)

        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame(columns=cols)

    def get_top_lab_items(self, lab_df, top_k=200):
        print(f"Selecting top {top_k} most frequent lab itemids...")
        counts = lab_df["itemid"].value_counts()
        top_items = set(counts.head(top_k).index)
        return top_items

    def get_top_items(self, df, col, top_k=20):
        """
        Generic helper to get top-k frequent values from a column.
        Used for procedureevents.itemid and emar.medication.
        """
        print(f"Selecting top {top_k} most frequent values in {col}...")
        counts = df[col].value_counts()
        return set(counts.head(top_k).index)

    def format_value(self, val, decimals=2):
        """
        Convert numeric value to a digit-level string.

        Example:
            1.23  -> "1 . 2 3"
            50    -> "5 0"
            49.999998 -> "5 0"
        """
        try:
            v = float(val)
        except (TypeError, ValueError):
            # Non-numeric fallback: just split chars
            s = str(val)
            return " ".join(list(s))

        # Format with fixed decimals, then strip trailing zeros and dot
        s = f"{v:.{decimals}f}"          # e.g. "49.99", "50.00"
        s = s.rstrip("0").rstrip(".")    # "50.00" -> "50", "3.40" -> "3.4"
        return " ".join(list(s))

    def get_time_tokens(self, dt, intime):
        """
        Convert absolute timestamp to relative time tokens:
          [DAYd] [WEEKDAY] [HHh] [MMm]
        where minutes are bucketed into 10-min bins.
        """
        delta = dt - intime
        total_minutes = int(delta.total_seconds() // 60)
        if total_minutes < 0:
            total_minutes = 0
        hours_total = total_minutes // 60
        minutes = total_minutes % 60

        day = hours_total // 24
        hour = hours_total % 24
        minute_bin = (minutes // 10) * 10  # 0,10,20,30,40,50
        weekday_idx = dt.weekday()  # 0=Mon, 6=Sun

        key = (day, weekday_idx, hour, minute_bin)
        if key in self.time_cache:
            return self.time_cache[key]

        d_token = f"[DAY{min(day, 3)}]"
        w_token = self.weekdays[weekday_idx]
        h_token = f"[{hour:02d}h]"
        m_token = f"[{minute_bin:02d}m]"

        tokens = f"{d_token} {w_token} {h_token} {m_token}"
        self.time_cache[key] = tokens
        return tokens

    # ---------------------------------------------------------------------
    # Main preprocessing pipeline
    # ---------------------------------------------------------------------
    def process_data(self):
        """
        Main pipeline:
          1. Load ICU stays (cohort).
          2. Load events:
             - labevents (top-200 itemid)
             - inputevents
             - outputevents
             - procedureevents (top-20 itemid)
             - EMAR + EMAR_DETAIL
          3. Merge demographics.
          4. Restrict to first 72 hours of ICU stay.
        Returns:
            all_data: list of dict(stay_id, demographics, events, intime)
            item_map: dict from itemid/medication string to human-readable label
        """
        # 1. ICU stays
        print("Loading ICU stays...")
        icu_df = self.load_icustays()
        icu_df = icu_df.dropna(subset=["intime", "outtime"])
        icu_df = icu_df.sort_values(["subject_id", "intime"])

        if self.stay_limit:
            print(f"Limiting to {self.stay_limit} ICU stays...")
            icu_df = icu_df.head(self.stay_limit)

        hadm_ids = set(icu_df["hadm_id"].unique())

        # ---------------------------------------------------------------
        # 2. Lab events (top-200 itemid)
        # ---------------------------------------------------------------
        lab_cols = [
            "subject_id",
            "hadm_id",
            "itemid",
            "charttime",
            "valuenum",
            "valueuom",
        ]
        lab_raw = self.load_events_chunked(
            "labevents.csv.gz", lab_cols, "charttime", hadm_ids
        )
        lab_raw = lab_raw.dropna(subset=["valuenum", "charttime"])

        top_labs = self.get_top_lab_items(lab_raw, top_k=200)
        print(f"Number of lab events before top-k filtering: {len(lab_raw)}")
        print(f"Top lab itemids: {sorted(list(top_labs))[:20]} ...")
        lab_df = lab_raw[lab_raw["itemid"].isin(top_labs)].copy()
        lab_df["event_type"] = "labevent"
        lab_df = lab_df.rename(
            columns={"charttime": "time", "valuenum": "value", "valueuom": "uom"}
        )

        # ---------------------------------------------------------------
        # 3. Input events (fluid / medication given in ICU)
        # ---------------------------------------------------------------
        input_cols = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "itemid",
            "starttime",
            "amount",
            "amountuom",
        ]
        input_df = self.load_events_chunked(
            "inputevents.csv.gz", input_cols, "starttime", hadm_ids
        )
        input_df = input_df.dropna(subset=["amount", "starttime"])
        input_df["event_type"] = "inputevent"
        input_df = input_df.rename(
            columns={"starttime": "time", "amount": "value", "amountuom": "uom"}
        )

        # ---------------------------------------------------------------
        # 4. Output events (urine output, drains, etc.)
        # ---------------------------------------------------------------
        output_cols = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "itemid",
            "charttime",
            "value",
            "valueuom",
        ]
        output_df = self.load_events_chunked(
            "outputevents.csv.gz", output_cols, "charttime", hadm_ids
        )
        output_df = output_df.dropna(subset=["value", "charttime"])
        output_df["event_type"] = "outputevent"
        output_df = output_df.rename(columns={"charttime": "time", "valueuom": "uom"})

        # ---------------------------------------------------------------
        # 5. Procedure events (top-20 itemid)
        # ---------------------------------------------------------------
        proc_cols = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "itemid",
            "starttime",
            "value",
            "valueuom",
        ]
        print("Loading procedureevents...")
        proc_raw = self.load_events_chunked(
            "procedureevents.csv.gz", proc_cols, "starttime", hadm_ids
        )
        proc_raw = proc_raw.dropna(subset=["value", "starttime"])

        if not proc_raw.empty:
            proc_counts = proc_raw["itemid"].value_counts()
            top_proc_items = set(proc_counts.head(20).index)
            print(f"Selecting top 20 procedure itemids: {sorted(list(top_proc_items))}")
            proc_df = proc_raw[proc_raw["itemid"].isin(top_proc_items)].copy()
        else:
            print("No procedureevents found after filtering hadm_id.")
            proc_df = pd.DataFrame(columns=proc_cols)

        if not proc_df.empty:
            proc_df["event_type"] = "procedureevent"
            proc_df = proc_df.rename(
                columns={"starttime": "time", "valueuom": "uom"}
            )

        # ---------------------------------------------------------------
        # 6. EMAR + EMAR_DETAIL: medication administration
        # ---------------------------------------------------------------
        print("Loading EMAR + EMAR_DETAIL...")

        emar_path = os.path.join(self.data_dir, "emar.csv.gz")
        emar_cols = [
            "subject_id",
            "hadm_id",
            "emar_id",
            "charttime",
            "medication",
        ]

        emar_chunks = []
        if os.path.exists(emar_path):
            print(f"Reading emar.csv.gz from {emar_path}...")
            for chunk in pd.read_csv(emar_path, usecols=emar_cols, chunksize=1_000_000):
                chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
                if chunk.empty:
                    continue
                chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
                chunk = chunk.dropna(subset=["charttime"])
                if chunk.empty:
                    continue
                emar_chunks.append(chunk)

        if emar_chunks:
            emar_df = pd.concat(emar_chunks, ignore_index=True)
        else:
            emar_df = pd.DataFrame(columns=emar_cols)

        # emar_detail for dose_given, dose_given_unit
        detail_path = os.path.join(self.data_dir, "emar_detail.csv.gz")
        detail_cols = ["emar_id", "dose_given", "dose_given_unit"]

        if os.path.exists(detail_path) and not emar_df.empty:
            print(f"Reading emar_detail.csv.gz from {detail_path}...")
            detail_chunks = []
            emar_ids = set(emar_df["emar_id"].unique())

            for chunk in pd.read_csv(
                detail_path, usecols=detail_cols, chunksize=1_000_000
            ):
                chunk = chunk[chunk["emar_id"].isin(emar_ids)]
                if chunk.empty:
                    continue
                detail_chunks.append(chunk)

            if detail_chunks:
                emar_detail = pd.concat(detail_chunks, ignore_index=True)
            else:
                emar_detail = pd.DataFrame(columns=detail_cols)

            emar_df = emar_df.merge(emar_detail, on="emar_id", how="left")
        else:
            if not os.path.exists(detail_path):
                print(
                    "WARNING: emar_detail.csv.gz not found, dose information will be missing."
                )
            if not emar_df.empty:
                emar_df["dose_given"] = np.nan
                emar_df["dose_given_unit"] = np.nan

        if not emar_df.empty:
            emar_df["event_type"] = "emarevent"
            emar_df = emar_df.rename(columns={"charttime": "time"})
            emar_df["value"] = emar_df["dose_given"]
            emar_df["uom"] = emar_df["dose_given_unit"]
            emar_df["medication"] = emar_df["medication"].astype(str)
            emar_df["itemid"] = emar_df["medication"]
        else:
            emar_df = pd.DataFrame(
                columns=[
                    "subject_id",
                    "hadm_id",
                    "emar_id",
                    "time",
                    "medication",
                    "value",
                    "uom",
                    "itemid",
                    "event_type",
                ]
            )

        # ---------------------------------------------------------------
        # 7. Item dictionaries
        # ---------------------------------------------------------------
        print("Loading item dictionary tables...")
        d_lab = pd.read_csv(
            os.path.join(self.data_dir, "d_labitems.csv.gz"),
            usecols=["itemid", "label"],
        )
        d_items = pd.read_csv(
            os.path.join(self.data_dir, "d_items.csv.gz"),
            usecols=["itemid", "label"],
        )

        item_map = {}
        for _, row in d_lab.iterrows():
            item_map[row["itemid"]] = str(row["label"]).lower()
        for _, row in d_items.iterrows():
            item_map[row["itemid"]] = str(row["label"]).lower()

        # EMAR medication string
        if not emar_df.empty:
            unique_meds = emar_df["medication"].dropna().unique()
            for med in unique_meds:
                item_map[med] = str(med).lower()

        # ---------------------------------------------------------------
        # 8. Demographics
        # ---------------------------------------------------------------
        print("Loading patient demographics...")
        pat_df = pd.read_csv(
            os.path.join(self.data_dir, "patients.csv.gz"),
            usecols=["subject_id", "gender", "anchor_age"],
        )
        adm_df = pd.read_csv(
            os.path.join(self.data_dir, "admissions.csv.gz"),
            usecols=["subject_id", "hadm_id", "race"],
        )

        icu_df = icu_df.merge(pat_df, on="subject_id", how="left")
        icu_df = icu_df.merge(adm_df[["hadm_id", "race"]], on="hadm_id", how="left")

        # ---------------------------------------------------------------
        # 9. Merge events per stay, filter to first 72h of ICU stay
        # ---------------------------------------------------------------
        print("Merging and filtering events within 72h window...")

        all_data = []

        lab_grouped = lab_df.groupby("hadm_id") if not lab_df.empty else {}
        input_grouped = input_df.groupby("stay_id") if not input_df.empty else {}
        output_grouped = output_df.groupby("stay_id") if not output_df.empty else {}
        proc_grouped = proc_df.groupby("stay_id") if not proc_df.empty else {}
        emar_grouped = emar_df.groupby("hadm_id") if not emar_df.empty else {}

        for _, stay in tqdm(
            icu_df.iterrows(), total=len(icu_df), desc="Processing stays"
        ):
            stay_id = stay["stay_id"]
            hadm_id = stay["hadm_id"]
            intime = stay["intime"]
            cutoff = intime + pd.Timedelta(hours=72)

            stay_events = []

            # Labs (hadm_id)
            if hasattr(lab_grouped, "groups") and hadm_id in lab_grouped.groups:
                labs = lab_grouped.get_group(hadm_id)
                mask = (labs["time"] >= intime) & (labs["time"] <= cutoff) & (
                    labs["time"] <= stay["outtime"]
                )
                labs = labs[mask]
                if not labs.empty:
                    stay_events.append(labs)

            # Input events (stay_id)
            if hasattr(input_grouped, "groups") and stay_id in input_grouped.groups:
                inputs = input_grouped.get_group(stay_id)
                mask = (inputs["time"] >= intime) & (inputs["time"] <= cutoff)
                inputs = inputs[mask]
                if not inputs.empty:
                    stay_events.append(inputs)

            # Output events (stay_id)
            if hasattr(output_grouped, "groups") and stay_id in output_grouped.groups:
                outputs = output_grouped.get_group(stay_id)
                mask = (outputs["time"] >= intime) & (outputs["time"] <= cutoff)
                outputs = outputs[mask]
                if not outputs.empty:
                    stay_events.append(outputs)

            # Procedure events (stay_id)
            if hasattr(proc_grouped, "groups") and stay_id in proc_grouped.groups:
                procs = proc_grouped.get_group(stay_id)
                mask = (procs["time"] >= intime) & (procs["time"] <= cutoff)
                procs = procs[mask]
                if not procs.empty:
                    stay_events.append(procs)

            # EMAR events (hadm_id)
            if hasattr(emar_grouped, "groups") and hadm_id in emar_grouped.groups:
                emars = emar_grouped.get_group(hadm_id)
                mask = (emars["time"] >= intime) & (emars["time"] <= cutoff) & (
                    emars["time"] <= stay["outtime"]
                )
                emars = emars[mask]
                if not emars.empty:
                    stay_events.append(emars)

            if not stay_events:
                continue

            combined = pd.concat(stay_events, ignore_index=True)
            if combined.empty:
                continue

            combined = combined.sort_values("time")

            all_data.append(
                {
                    "stay_id": stay_id,
                    "demographics": stay,
                    "events": combined,
                    "intime": intime,
                }
            )

        print(f"Total stays after filtering: {len(all_data)}")

        if self.inspect:
            self.inspect_events(all_data, item_map)

        return all_data, item_map

    # ---------------------------------------------------------------------
    # Inspect: print a few textual events
    # ---------------------------------------------------------------------
    def inspect_events(self, data, item_map):
        print("\n=== Inspect mode: printing a few raw textual events ===")
        n_stays = min(self.inspect_max_stays, len(data))
        for i in range(n_stays):
            d = data[i]
            stay_id = d["stay_id"]
            demo = d["demographics"]
            events = d["events"]
            intime = d["intime"]

            gender = str(demo["gender"]).lower()
            age = str(demo["anchor_age"])
            race = str(demo["race"]).lower()
            demo_text = f"gender {gender} age {age} race {race}"

            print(f"\n--- Stay {i} (stay_id={stay_id}) ---")
            print(f"Demographics: {demo_text}")

            sub_events = events.head(self.inspect_max_events)
            for _, ev in sub_events.iterrows():
                t_str = self.get_time_tokens(ev["time"], intime)
                e_type = ev["event_type"]

                if e_type == "emarevent":
                    item_name = str(ev.get("medication", "unknown")).lower()
                else:
                    item_name = item_map.get(ev.get("itemid", None), "unknown")

                val_str = self.format_value(ev["value"])
                uom = str(ev["uom"]).lower() if pd.notna(ev["uom"]) else ""
                text = f"{t_str} {e_type} {item_name} {val_str} {uom} [EOE]"
                print(f"Event @ {ev['time']}: {text}")
        print("=== End of inspect ===\n")

    # ---------------------------------------------------------------------
    # Sequence construction
    # ---------------------------------------------------------------------
    def create_sequences(self, data, item_map, split_name):
        sequences = []
        eval_items = []
        shard_idx = 0

        print(f"Creating token sequences for split = {split_name}...")

        for item in tqdm(data):
            stay_id = item["stay_id"]
            demo = item["demographics"]
            events = item["events"]
            intime = item["intime"]

            gender = str(demo["gender"]).lower()
            age = str(demo["anchor_age"])
            race = str(demo["race"]).lower()
            demo_text = f"gender {gender} age {age} race {race}"
            demo_tokens = self.tokenizer.encode(demo_text, add_special_tokens=False)

            current_tokens = demo_tokens.copy()
            current_type_ids = [0] * len(demo_tokens)  # 0 = never predict

            for _, ev in events.iterrows():
                t_str = self.get_time_tokens(ev["time"], intime)
                t_tokens = self.tokenizer.encode(t_str, add_special_tokens=False)

                e_type = ev["event_type"]

                if e_type == "emarevent":
                    item_name = str(ev.get("medication", "unknown")).lower()
                else:
                    item_name = item_map.get(ev.get("itemid", None), "unknown")

                val_str = self.format_value(ev["value"])
                uom = str(ev["uom"]).lower() if pd.notna(ev["uom"]) else ""

                type_token_id = self.tokenizer.convert_tokens_to_ids(e_type)

                item_tokens = self.tokenizer.encode(item_name, add_special_tokens=False)
                val_uom_text = f"{val_str} {uom} [EOE]"
                val_tokens = self.tokenizer.encode(val_uom_text, add_special_tokens=False)

                event_full_tokens = t_tokens + [type_token_id] + item_tokens + val_tokens

                # type_ids:
                #   0 for everything except lab values
                #   1 for val+uom+[EOE] of lab events (target tokens)
                time_type_ids = [0] * len(t_tokens)
                etype_type_ids = [0]
                item_type_ids = [0] * len(item_tokens)
                if e_type == "labevent":
                    val_type_ids = [1] * len(val_tokens)
                else:
                    val_type_ids = [0] * len(val_tokens)

                event_type_ids = (
                    time_type_ids + etype_type_ids + item_type_ids + val_type_ids
                )

                # Check length
                if len(current_tokens) + len(event_full_tokens) > self.max_len:
                    sequences.append(
                        {
                            "stay_id": stay_id,
                            "input_ids": current_tokens,
                            "type_ids": current_type_ids,
                        }
                    )
                    current_tokens = demo_tokens.copy()
                    current_type_ids = [0] * len(demo_tokens)

                # For eval: only create targets for lab events
                if split_name in ["val", "test"] and e_type == "labevent":
                    prompt_ids = (
                        current_tokens + t_tokens + [type_token_id] + item_tokens
                    )
                    label_ids = val_tokens

                    if len(prompt_ids) > self.max_len:
                        available = self.max_len - len(demo_tokens)
                        tail = prompt_ids[-available:]
                        prompt_ids = demo_tokens + tail

                    eval_items.append(
                        {
                            "stay_id": stay_id,
                            "prompt_ids": prompt_ids,
                            "label_ids": label_ids,
                            "valuenum": ev["value"],
                            "itemid": ev.get("itemid", None),
                            "event_type": e_type,
                        }
                    )

                current_tokens.extend(event_full_tokens)
                current_type_ids.extend(event_type_ids)

            if len(current_tokens) > len(demo_tokens):
                sequences.append(
                    {
                        "stay_id": stay_id,
                        "input_ids": current_tokens,
                        "type_ids": current_type_ids,
                    }
                )

            if len(sequences) >= self.shard_size:
                self._write_shard(sequences, split_name, shard_idx)
                sequences = []
                shard_idx += 1

        if sequences:
            self._write_shard(sequences, split_name, shard_idx)
            shard_idx += 1

        self._merge_shards(split_name, shard_idx)

        if eval_items and split_name in ["val", "test"]:
            with open(os.path.join(self.out_dir, f"{split_name}_eval.pkl"), "wb") as f:
                pickle.dump(eval_items, f)

    # ---------------------------------------------------------------------
    # Shard I/O
    # ---------------------------------------------------------------------
    def _write_shard(self, data, split_name, shard_idx):
        filename = os.path.join(self.out_dir, f"{split_name}_shard_{shard_idx}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def _merge_shards(self, split_name, num_shards):
        print(f"Merging {num_shards} shards for split = {split_name}...")
        all_data = []
        for i in range(num_shards):
            filename = os.path.join(self.out_dir, f"{split_name}_shard_{i}.pkl")
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    shard_data = pickle.load(f)
                    all_data.extend(shard_data)
                os.remove(filename)
        with open(os.path.join(self.out_dir, f"{split_name}.pkl"), "wb") as f:
            pickle.dump(all_data, f)

    # ---------------------------------------------------------------------
    # Train/val/test split + run
    # ---------------------------------------------------------------------
    def run(self):
        all_data, item_map = self.process_data()
        print(f"Selected {len(all_data)} stays for processing.")

        subjects = list(set(d["demographics"]["subject_id"] for d in all_data))
        train_subs, test_subs = train_test_split(
            subjects, test_size=0.2, random_state=42
        )
        val_subs, test_subs = train_test_split(
            test_subs, test_size=0.5, random_state=42
        )

        train_data = [d for d in all_data if d["demographics"]["subject_id"] in train_subs]
        val_data = [d for d in all_data if d["demographics"]["subject_id"] in val_subs]
        test_data = [d for d in all_data if d["demographics"]["subject_id"] in test_subs]

        print(
            f"Train stays: {len(train_data)}, "
            f"Val stays: {len(val_data)}, "
            f"Test stays: {len(test_data)}"
        )

        self.create_sequences(train_data, item_map, "train")
        self.create_sequences(val_data, item_map, "val")
        self.create_sequences(test_data, item_map, "test")

        print("Preprocessing complete.")

    def load_subjects(self, limit=None):
        # Placeholder to align with some frameworks that expect this method.
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-IV data into LabTOP-style token sequences."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the MIMIC-IV data directory (with icustays.csv.gz, labevents.csv.gz, etc.).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="processed_data",
        help="Directory to save processed tokenizer and .pkl files.",
    )
    parser.add_argument(
        "--stay_limit",
        type=int,
        default=None,
        help="Optional limit on number of ICU stays for a quick demo.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=5000,
        help="Number of sequences per temporary shard before merging.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="If set, print a few example events as raw text.",
    )
    parser.add_argument(
        "--inspect_max_stays",
        type=int,
        default=5,
        help="Number of stays to print in inspect mode.",
    )
    parser.add_argument(
        "--inspect_max_events",
        type=int,
        default=10,
        help="Number of events per stay to print in inspect mode.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    preprocessor = LabTOPPreprocessor(
        data_dir=args.data_dir,
        max_len=1024,
        stay_limit=args.stay_limit,
        shard_size=args.shard_size,
        inspect=args.inspect,
        inspect_max_stays=args.inspect_max_stays,
        inspect_max_events=args.inspect_max_events,
        out_dir=args.out_dir,
    )
    preprocessor.run()