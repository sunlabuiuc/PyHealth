"""Tests for MEME (Lee et al.), MIMIC-IV-ED loading and the ED pseudo-note tasks."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import yaml
from transformers import BertConfig, BertModel, BertTokenizerFast

import pyhealth.datasets
from pyhealth.datasets import (
    MIMIC4Dataset,
    MIMIC4EDDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.models import MEME, DomainSelfAttention, LLMEHREncoder
from pyhealth.tasks import EDDispositionMIMIC4, EDOutcomesMIMIC4
from pyhealth.tasks.ed_disposition_mimic4 import (
    arrival_note,
    codes_note,
    medrecon_note,
    pyxis_note,
    triage_note,
    vitals_note,
)
from pyhealth.trainer import Trainer

# Tiny fake MIMIC-IV-ED and MIMIC-IV tables. Patient 1 has two ED stays
# (admitted, then home); patient 2 was transferred (no label) and then
# admitted to the ICU and died; patient 3 has no hospital record.
ED_TABLES = {
    "ed/edstays.csv": [
        ("subject_id,hadm_id,stay_id,intime,outtime,gender,race,"
         "arrival_transport,disposition"),
        "1,100,11,2150-01-01 10:00:00,2150-01-01 18:00:00,F,WHITE,AMBULANCE,ADMITTED",
        "1,,12,2150-06-01 08:00:00,2150-06-01 12:00:00,F,WHITE,WALK IN,HOME",
        "2,,21,2160-03-01 09:00:00,2160-03-01 10:00:00,M,ASIAN,WALK IN,TRANSFER",
        "2,200,22,2160-04-01 09:00:00,2160-04-01 20:00:00,M,ASIAN,AMBULANCE,ADMITTED",
        "3,,31,2170-01-01 09:00:00,2170-01-01 11:00:00,M,BLACK,WALK IN,HOME",
    ],
    "ed/triage.csv": [
        ("subject_id,stay_id,temperature,heartrate,resprate,o2sat,sbp,dbp,pain,"
         "acuity,chiefcomplaint"),
        "1,11,98.4,110,22,91,90,60,8,2,Chest pain",
        "1,12,98.6,70,16,99,120,80,0,4,Rash",
        "2,22,,,,,,,,1,Unresponsive",
    ],
    "ed/vitalsign.csv": [
        ("subject_id,stay_id,charttime,temperature,heartrate,resprate,o2sat,sbp,"
         "dbp,rhythm,pain"),
        "1,11,2150-01-01 11:00:00,98.5,105,20,93,95,62,,7",
        "1,11,2150-01-01 17:00:00,98.7,90,18,96,110,70,,3",
        "1,12,2150-06-01 09:00:00,98.6,72,16,99,121,79,,0",
    ],
    "ed/medrecon.csv": [
        "subject_id,stay_id,charttime,name,gsn,ndc,etc_rn,etccode,etcdescription",
        "1,11,2150-01-01 10:30:00,Lisinopril,1,2,1,3,ACE Inhibitors",
    ],
    "ed/pyxis.csv": [
        "subject_id,stay_id,charttime,med_rn,name,gsn_rn,gsn",
        "1,11,2150-01-01 11:30:00,1,Aspirin,1,1",
        "1,11,2150-01-01 11:30:00,2,Nitroglycerin,1,2",
        "1,11,2150-01-01 16:00:00,3,Heparin,1,3",
        "2,22,2160-04-01 10:00:00,1,Naloxone,1,4",
    ],
    "ed/diagnosis.csv": [
        "subject_id,stay_id,seq_num,icd_code,icd_version,icd_title",
        "1,11,2,I10,10,ESSENTIAL HYPERTENSION",
        "1,11,1,I21.4,10,NSTEMI",
        "2,22,1,T40.2X1A,10,OPIOID POISONING",
    ],
}
EHR_TABLES = {
    "hosp/patients.csv": [
        "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod",
        "1,F,60,2148,2008 - 2010,",
        "2,M,40,2160,2014 - 2016,2160-04-03",
    ],
    "hosp/admissions.csv": [
        ("subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,"
         "admit_provider_id,admission_location,discharge_location,insurance,"
         "language,marital_status,race,edregtime,edouttime,hospital_expire_flag"),
        ("1,100,2150-01-01 18:00:00,2150-01-05 12:00:00,,URGENT,P1,EMERGENCY ROOM,"
         "HOME,Medicare,ENGLISH,MARRIED,WHITE,,,0"),
        ("2,200,2160-04-01 20:00:00,2160-04-03 12:00:00,2160-04-03 12:00:00,URGENT,"
         "P2,EMERGENCY ROOM,DIED,Other,ENGLISH,SINGLE,ASIAN,,,1"),
    ],
    "icu/icustays.csv": [
        "subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los",
        "2,200,9000,MICU,MICU,2160-04-01 21:00:00,2160-04-03 12:00:00,1.6",
    ],
}
NOTE_TABLES = {
    "note/discharge.csv": [
        "note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,text",
        "1-DS-1,1,100,DS,1,2150-01-05 00:00:00,2150-01-05 12:00:00,Discharged home.",
    ],
}


def _write(root, tables):
    for rel, lines in tables.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")


class TestNotes(unittest.TestCase):
    def test_arrival(self):
        self.assertEqual(
            arrival_note("M", "ASIAN", "WALK IN", "2160-04-01 09:00:00", 40),
            "A 40 year old asian male, arrived via walk in at 2160-04-01 09:00:00.",
        )
        self.assertEqual(
            arrival_note("F", None, "AMBULANCE", "t"),
            "A female, arrived via ambulance at t.",
        )
        for missing in (float("nan"), "nan", ""):
            self.assertEqual(arrival_note("F", missing, "AMBULANCE", "t", 30),
                             "A 30 year old female, arrived via ambulance at t.")

    def test_triage_and_missing_values(self):
        self.assertEqual(triage_note(None), "Triage measurements were not taken.")
        t = SimpleNamespace(temperature=None, heartrate="nan", resprate="16",
                            o2sat="97", sbp="106", dbp="63", pain="0",
                            acuity="3", chiefcomplaint="Fall")
        self.assertEqual(triage_note(t), (
            "At triage: temperature was not recorded, pulse was not recorded, "
            "respirations was 16, o2 saturation was 97, systolic blood pressure "
            "was 106, diastolic blood pressure was 63, pain was 0, chief "
            "complaint was fall. Acuity score was 3."))

    def test_lists(self):
        v = SimpleNamespace(timestamp="t1", temperature="98", heartrate="80",
                            resprate="16", o2sat="99", sbp="120", dbp="80",
                            pain="0")
        self.assertEqual(vitals_note([v]), (
            "The patient had the following vitals: At t1, temperature was 98, "
            "pulse was 80, respirations was 16, o2 saturation was 99, systolic "
            "blood pressure was 120, diastolic blood pressure was 80, pain was 0."))
        doses = [SimpleNamespace(timestamp="t1", name="Aspirin"),
                 SimpleNamespace(timestamp="t1", name="Heparin"),
                 SimpleNamespace(timestamp="t2", name="Morphine")]
        self.assertEqual(pyxis_note(doses), (
            "The patient received the following medications: At t1, aspirin, "
            "heparin were administered. At t2, morphine were administered."))
        self.assertEqual(pyxis_note([]), "The patient did not receive any medications.")
        self.assertEqual(
            medrecon_note([SimpleNamespace(name="Lisinopril",
                                           etcdescription="ACE Inhibitors")]),
            "The patient was previously taking the following medications: "
            "lisinopril, ace inhibitors.")
        self.assertEqual(
            codes_note([SimpleNamespace(icd_version="10", icd_code="I21.4",
                                        icd_title="NSTEMI")]),
            "The patient received the following diagnostic codes: ICD-10 code: "
            "[i214], nstemi.")
        self.assertEqual(codes_note([]), "The patient had no diagnostic codes recorded.")


class TestEDTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls._tmp.name
        _write(os.path.join(tmp, "ed"), ED_TABLES)
        _write(os.path.join(tmp, "ehr"), EHR_TABLES)
        cls.dataset = MIMIC4Dataset(
            ehr_root=os.path.join(tmp, "ehr"),
            ed_root=os.path.join(tmp, "ed"),
            ed_tables=["triage", "vitalsign", "medrecon", "pyxis", "diagnosis"],
            cache_dir=os.path.join(tmp, "cache"),
        )
        cls.patients = {p.patient_id: p for p in cls.dataset.iter_patients()}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _samples(self, task):
        return {s["visit_id"]: s for p in self.patients.values() for s in task(p)}

    def test_disposition_labels_and_notes(self):
        task = EDDispositionMIMIC4()
        self.assertEqual(list(task.input_schema),
                         ["arrival", "triage", "medrecon", "vitals", "pyxis"])
        samples = self._samples(task)
        # the transfer (stay 21) has no label
        self.assertEqual({k: s["disposition"] for k, s in samples.items()},
                         {"11": 1, "12": 0, "22": 1, "31": 0})
        first = samples["11"]
        self.assertEqual(first["patient_id"], "1")
        self.assertNotIn("codes", first)
        # age at the visit: anchor_age 60 in 2148, visit in 2150
        self.assertEqual(first["arrival"], "A 62 year old white female, arrived "
                         "via ambulance at 2150-01-01 10:00:00.")
        self.assertIn("chief complaint was chest pain", first["triage"])
        self.assertIn("lisinopril, ace inhibitors", first["medrecon"])
        # only this stay's events, grouped by time
        self.assertEqual(first["vitals"].count("At "), 2)
        self.assertNotIn("72", first["vitals"])
        self.assertIn("aspirin, nitroglycerin were administered", first["pyxis"])
        self.assertIn("heparin", first["pyxis"])
        # missing domains get the fixed texts; no patients row means no age
        self.assertEqual(samples["31"]["triage"], "Triage measurements were not taken.")
        self.assertEqual(samples["31"]["vitals"], "The patient had no vitals recorded.")
        self.assertTrue(samples["31"]["arrival"].startswith("A black male,"))

    def test_codes_are_optional_and_ordered(self):
        samples = self._samples(EDDispositionMIMIC4(include_codes=True))
        self.assertEqual(samples["11"]["codes"], (
            "The patient received the following diagnostic codes: ICD-10 code: "
            "[i214], nstemi. ICD-10 code: [i10], essential hypertension."))
        self.assertEqual(samples["12"]["codes"],
                         "The patient had no diagnostic codes recorded.")

    def test_observation_window(self):
        samples = self._samples(EDDispositionMIMIC4(observation_hours=2))
        first = samples["11"]
        self.assertEqual(first["vitals"].count("At "), 1)  # 11:00 kept, 17:00 not
        self.assertNotIn("heparin", first["pyxis"])  # 16:00 dropped
        self.assertIn("aspirin", first["pyxis"])

    def test_outcomes(self):
        task = EDOutcomesMIMIC4()
        samples = self._samples(task)
        self.assertEqual({k: s["outcomes"] for k, s in samples.items()},
                         {"11": [], "22": ["discharge_not_home", "mortality", "icu"]})
        self.assertEqual(task.output_schema, {"outcomes": "multilabel"})

    def test_outcomes_warn_without_ehr_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = MIMIC4Dataset(ed_root=os.path.join(self._tmp.name, "ed"),
                                    cache_dir=tmp)
            with self.assertLogs("pyhealth.tasks.ed_disposition_mimic4",
                                 "WARNING") as logs:
                EDOutcomesMIMIC4().pre_filter(dataset.global_event_df)
            self.assertIn("ehr_root", logs.output[0])
        with self.assertNoLogs("pyhealth.tasks.ed_disposition_mimic4", "WARNING"):
            EDOutcomesMIMIC4().pre_filter(self.dataset.global_event_df)

    def test_warn_about_unloaded_ed_tables(self):
        logger = "pyhealth.tasks.ed_disposition_mimic4"
        with tempfile.TemporaryDirectory() as tmp:
            dataset = MIMIC4Dataset(ed_root=os.path.join(self._tmp.name, "ed"),
                                    ed_tables=["triage"], cache_dir=tmp)
            df = dataset.global_event_df
            with self.assertLogs(logger, "WARNING") as logs:
                EDDispositionMIMIC4(include_codes=True).pre_filter(df)
            self.assertEqual(len(logs.output), 1)
            message = logs.output[0]
            for table in ("vitalsign", "medrecon", "pyxis", "diagnosis"):
                self.assertIn(f"'{table}'", message)
            self.assertNotIn("'triage'", message)
            self.assertIn("not recorded", message)
            # diagnosis is only needed with include_codes=True
            with self.assertLogs(logger, "WARNING") as logs:
                EDDispositionMIMIC4().pre_filter(df)
            self.assertNotIn("diagnosis", logs.output[0])
        for task in (EDDispositionMIMIC4(include_codes=True),
                     EDOutcomesMIMIC4(include_codes=True)):
            with self.assertNoLogs(logger, "WARNING"):
                task.pre_filter(self.dataset.global_event_df)

    def test_ed_dataset_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = MIMIC4EDDataset(root=os.path.join(self._tmp.name, "ed"),
                                      tables=["triage"], cache_dir=tmp)
            self.assertIn("edstays", dataset.tables)
            samples = [s for p in dataset.iter_patients()
                       for s in EDDispositionMIMIC4()(p)]
            self.assertEqual(len(samples), 4)
            self.assertTrue(all(s["arrival"].startswith("A ") and "year old"
                                not in s["arrival"] for s in samples))
            self.assertTrue(all(s["vitals"] == "The patient had no vitals recorded."
                                for s in samples))

    def test_set_task_and_train(self):
        samples = self.dataset.set_task(EDDispositionMIMIC4(include_codes=True),
                                        num_workers=1)
        self.assertEqual(len(samples), 4)
        with tempfile.TemporaryDirectory() as tmp:
            encoder, tokenizer = _tiny_encoder(tmp)
            model = MEME(samples, encoder=encoder, tokenizer=tokenizer)
            self.assertEqual(model.domains, ["arrival", "triage", "medrecon",
                                             "vitals", "pyxis", "codes"])
            trainer = Trainer(model=model, metrics=["roc_auc"], enable_logging=False)
            loader = get_dataloader(samples, batch_size=2, shuffle=True)
            trainer.train(train_dataloader=loader, epochs=1)
            scores = trainer.evaluate(get_dataloader(samples, batch_size=4))
            self.assertIn("roc_auc", scores)


class TestMIMIC4DatasetEDRoot(unittest.TestCase):
    """The ``ed_root``, ``ed_tables`` and ``ed_config_path`` of MIMIC4Dataset."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls._tmp.name
        cls.ed = os.path.join(tmp, "ed")
        cls.ehr = os.path.join(tmp, "ehr")
        cls.note = os.path.join(tmp, "note")
        _write(cls.ed, ED_TABLES)
        _write(cls.ehr, EHR_TABLES)
        _write(cls.note, NOTE_TABLES)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _cache(self):
        return tempfile.mkdtemp(dir=self._tmp.name)

    def test_root_and_cache_key_unchanged_without_ed_root(self):
        cache = self._cache()
        plain = MIMIC4Dataset(ehr_root=self.ehr, cache_dir=cache)
        self.assertEqual(plain.root, f"{self.ehr}|None|None")
        self.assertEqual(plain.tables, [])
        self.assertNotIn("ed", plain.sub_datasets)
        # ED arguments without ed_root are ignored, so the key is the same
        ignored = MIMIC4Dataset(ehr_root=self.ehr, ed_tables=["triage"],
                                ed_config_path="unused.yaml", cache_dir=cache)
        self.assertEqual(ignored.root, plain.root)
        self.assertEqual(ignored.tables, plain.tables)
        self.assertEqual(ignored.cache_dir, plain.cache_dir)
        with_ed = MIMIC4Dataset(ehr_root=self.ehr, ed_root=self.ed, cache_dir=cache)
        self.assertEqual(with_ed.root, f"{self.ehr}|None|None|{self.ed}")
        self.assertNotEqual(with_ed.cache_dir, plain.cache_dir)

    def test_ed_root_alone(self):
        dataset = MIMIC4Dataset(ed_root=self.ed, ed_tables=["triage"],
                                cache_dir=self._cache())
        self.assertEqual(list(dataset.sub_datasets), ["ed"])
        self.assertIsInstance(dataset.sub_datasets["ed"], MIMIC4EDDataset)
        self.assertEqual(sorted(dataset.tables), ["edstays", "triage"])
        self.assertEqual(sorted(dataset.unique_patient_ids), ["1", "2", "3"])
        triage = dataset.get_patient("1").get_events(event_type="triage")
        self.assertEqual(len(triage), 2)

    def test_ed_config_path(self):
        default = os.path.join(os.path.dirname(pyhealth.datasets.__file__),
                               "configs", "mimic4_ed.yaml")
        with open(default) as f:
            config = yaml.safe_load(f)
        config["tables"] = {k: v for k, v in config["tables"].items()
                            if k in ("edstays", "triage")}
        path = os.path.join(self._tmp.name, "custom_ed.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        dataset = MIMIC4Dataset(ed_root=self.ed, ed_tables=["triage"],
                                ed_config_path=path, cache_dir=self._cache())
        self.assertEqual(set(dataset.sub_datasets["ed"].config.tables),
                         {"edstays", "triage"})
        self.assertEqual(len(dataset.get_patient("1").get_events("triage")), 2)

    def test_ed_with_notes_and_no_root(self):
        with self.assertRaises(ValueError):
            MIMIC4Dataset()
        with self.assertWarns(UserWarning):  # discharge notes are date-only
            dataset = MIMIC4Dataset(note_root=self.note, note_tables=["discharge"],
                                    ed_root=self.ed, cache_dir=self._cache())
        self.assertEqual(list(dataset.sub_datasets), ["note", "ed"])
        patient = dataset.get_patient("1")
        self.assertEqual(len(patient.get_events("discharge")), 1)
        self.assertEqual(len(patient.get_events("edstays")), 2)

    def test_ed_with_cxr(self):
        cxr = os.path.join(self._tmp.name, "cxr")
        os.makedirs(cxr, exist_ok=True)
        pd.DataFrame({
            "dicom_id": ["d1"], "subject_id": ["1"], "study_id": ["50"],
            "PerformedProcedureStepDescription": ["CHEST (PA AND LAT)"],
            "ViewPosition": ["PA"], "Rows": ["3056"], "Columns": ["2544"],
            "StudyDate": ["21500101"], "StudyTime": ["123.002"],
            "ProcedureCodeSequence_CodeMeaning": ["CHEST (PA AND LAT)"],
            "ViewCodeSequence_CodeMeaning": ["postero-anterior"],
            "PatientOrientationCodeSequence_CodeMeaning": ["Erect"],
        }).to_csv(os.path.join(cxr, "mimic-cxr-2.0.0-metadata.csv.gz"), index=False)
        dataset = MIMIC4Dataset(cxr_root=cxr, cxr_tables=["metadata"],
                                ed_root=self.ed, cache_dir=self._cache())
        self.assertEqual(list(dataset.sub_datasets), ["cxr", "ed"])
        (image,) = dataset.get_patient("1").get_events("metadata")
        self.assertTrue(image.image_path.endswith(
            os.path.join("files", "p1", "p1", "s50", "d1.jpg")))
        self.assertEqual(str(image.timestamp), "2150-01-01 00:01:23")


def _tiny_encoder(tmp):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", ".", ":", "a",
             "the", "patient", "was", "at", "triage", "acuity", "score", "pain",
             "arrived", "via", "ambulance", "walk", "in", "chest", "1", "2", "3"]
    path = os.path.join(tmp, "vocab.txt")
    with open(path, "w") as f:
        f.write("\n".join(vocab))
    tokenizer = BertTokenizerFast(vocab_file=path)
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=128)
    return BertModel(config).eval(), tokenizer


def _toy_dataset(output="binary"):
    labels = {"binary": [1, 0, 1], "multilabel": [["icu"], [], ["icu", "mortality"]]}
    samples = [
        {"patient_id": f"p{i}", "arrival": a, "triage": t, "y": y}
        for i, (a, t, y) in enumerate(zip(
            ["arrived via ambulance.", "arrived via walk in.", "arrived via ambulance."],
            ["acuity score was 1.", "acuity score was 3.", "chest pain."],
            labels[output],
        ))
    ]
    return create_sample_dataset(samples, input_schema={"arrival": "text",
                                                        "triage": "text"},
                                 output_schema={"y": output})


class TestMEME(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.encoder, self.tokenizer = _tiny_encoder(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _model(self, dataset, **kw):
        return MEME(dataset, encoder=self.encoder, tokenizer=self.tokenizer, **kw)

    def test_embeddings_are_the_cls_token(self):
        model = self._model(_toy_dataset())
        texts = ["arrived via ambulance.", "acuity score was 1 , pain 3 ."]
        batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        with torch.no_grad():
            expected = self.encoder(**batch).last_hidden_state[:, 0]
        torch.testing.assert_close(model.embed(texts), expected)

    def test_reuses_llm_ehr_encoder(self):
        model = self._model(_toy_dataset(), max_length=16)
        encoder = model.text_encoder
        self.assertIsInstance(encoder, LLMEHREncoder)
        self.assertEqual((encoder.pooling, encoder.normalize, encoder.max_length),
                         ("cls", False, 16))
        self.assertEqual(model.embedding_dim, encoder.embedding_dim)
        self.assertNotIn(encoder, list(model.modules()))
        texts = ["arrived via ambulance."]
        torch.testing.assert_close(model.embed(texts), encoder.embed(texts))

    def test_forward_with_and_without_labels(self):
        dataset = _toy_dataset()
        model = self._model(dataset)
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (3, 1))
        self.assertEqual(out["attention"].shape, (3, 2, 2))
        torch.testing.assert_close(out["attention"].sum(-1), torch.ones(3, 2))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(set(out), {"logit", "y_prob", "embed", "attention",
                                    "loss", "y_true"})
        unlabeled = {k: v for k, v in batch.items() if k != "y"}
        out = model(**unlabeled)
        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)

    def test_only_the_head_trains_and_is_saved(self):
        dataset = _toy_dataset()
        model = self._model(dataset)
        frozen = {k: v.clone() for k, v in self.encoder.state_dict().items()}
        self.assertTrue(all(k.startswith(("attention.", "fc.", "_dummy"))
                            for k in model.state_dict()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        before = model.fc.weight.detach().clone()
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        model.train()
        model(**batch)["loss"].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fc.weight))
        for k, v in self.encoder.state_dict().items():
            self.assertTrue(torch.equal(v, frozen[k]), k)

    def test_multilabel_and_mean_readout(self):
        dataset = _toy_dataset("multilabel")
        model = self._model(dataset, readout="mean", feature_keys=["triage"])
        self.assertEqual(model.domains, ["triage"])
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (3, 2))
        self.assertEqual(out["attention"].shape, (3, 1, 1))
        self.assertTrue(torch.isfinite(out["loss"]))
        with self.assertRaises(ValueError):
            self._model(dataset, readout="max")
        with self.assertRaises(ValueError):
            self._model(dataset, feature_keys=["nope"])

    def test_embeddings_are_cached(self):
        calls = []

        def embed_fn(texts):
            calls.append(list(texts))
            return torch.ones(len(texts), 4) * len(texts[0])

        dataset = _toy_dataset()
        cache = os.path.join(self._tmp.name, "cache")
        with self.assertRaises(ValueError):  # a custom embedder needs a name
            MEME(dataset, embed_fn=embed_fn, cache_dir=cache)
        model = MEME(dataset, embed_fn=embed_fn, cache_dir=cache, cache_name="t")
        calls.clear()  # the size probe
        first = model.embed(["a", "bb", "a"])
        self.assertEqual(calls, [["a", "bb"]])  # each text once
        model.embed(["bb"])
        self.assertEqual(len(calls), 1)  # memoized
        fresh = MEME(dataset, embed_fn=embed_fn, cache_dir=cache, cache_name="t")
        calls.clear()
        torch.testing.assert_close(fresh.embed(["a", "bb", "a"]), first)
        self.assertEqual(calls, [])  # read from disk

    def test_domain_self_attention(self):
        torch.manual_seed(0)
        layer = DomainSelfAttention(4, dropout=0.0)
        x = torch.randn(2, 3, 4)
        out, attention = layer(x)
        # permuting domains permutes the output: no positional terms
        perm = torch.tensor([2, 0, 1])
        out_p, _ = layer(x[:, perm])
        torch.testing.assert_close(out_p, out[:, perm])
        self.assertEqual(attention.shape, (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
