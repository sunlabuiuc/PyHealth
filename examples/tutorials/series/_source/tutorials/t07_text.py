from nbkit import code, footer, header, md

FILENAME = "07_clinical_text.ipynb"


def cells():
    return [
        *header(
            "07",
            "Clinical text: classification and medical coding",
            [
                "How to fine-tune a Hugging Face language model on clinical notes with PyHealth",
                "Multiclass text classification: which specialty wrote this report?",
                "Multilabel medical coding: which ICD codes does this note support?",
                "Why a simple TF-IDF baseline belongs next to every language model",
            ],
            25,
            "Tutorials 00 and 02. A GPU runtime (Runtime > Change runtime type > T4 GPU) makes training much faster.",
        ),
        md("""
        ## Text is just another field

        For PyHealth, a note is one more input field. The `text` processor
        keeps it as a string, and `TransformersModel` wraps any Hugging Face
        encoder: it tokenizes the text (first 256 tokens), encodes it, and
        adds a prediction head that matches the task's label (multiclass,
        multilabel, binary).

        Model size matters on a free runtime. We pick automatically: a
        clinical BERT (110M parameters, trained on MIMIC notes) for 3 epochs on
        a GPU, or a small general BERT (11M parameters) for 1 epoch on a CPU.
        **Use a GPU runtime if you can** (Runtime > Change runtime type > T4
        GPU): on a CPU even the small model needs about 10 minutes per epoch,
        and scores after one epoch are lower.
        """),
        code("""
        import torch

        if torch.cuda.is_available():
            MODEL_NAME, EPOCHS = "emilyalsentzer/Bio_ClinicalBERT", 3
        else:
            # About 10 minutes per epoch on a free CPU runtime; a GPU is much faster.
            MODEL_NAME, EPOCHS = "google/bert_uncased_L-4_H-256_A-4", 1
        print("device:", "cuda" if torch.cuda.is_available() else "cpu", "| model:", MODEL_NAME)
        """),
        md("""
        ## Part A. Which specialty wrote this report?

        The [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
        collection has about 5,000 transcribed medical reports, each labelled
        with a medical specialty. These are sample documents, not patient
        records, so each report is its own "patient".
        """),
        code("""
        import urllib.request
        import zipfile

        urllib.request.urlretrieve(
            "https://storage.googleapis.com/pyhealth/medical_transcriptions_data/MedicalTranscriptions.zip",
            "MedicalTranscriptions.zip",
        )
        zipfile.ZipFile("MedicalTranscriptions.zip").extractall(".")

        from pyhealth.datasets import MedicalTranscriptionsDataset

        mts = MedicalTranscriptionsDataset(root="MedicalTranscriptions", cache_dir="pyhealth_cache")
        mts.stats()
        """),
        md("""
        The dataset comes with a default task, text in and specialty out:
        """),
        code("""
        print(mts.default_task.input_schema, "->", mts.default_task.output_schema)
        reports = mts.set_task()

        sample = reports[0]
        print(f"{len(reports)} reports")
        print("text:", sample["transcription"][:300], "...")
        labels = reports.output_processors["medical_specialty"].label_vocab
        print(f"\\n{len(labels)} specialties, e.g. {list(labels)[:6]}")
        """),
        md("""
        Each report is its own patient, so a sample-level split is fine here.
        Then the usual PyHealth steps: model, trainer, evaluation. For
        multiclass labels the trainer reports accuracy and F1; macro F1
        weights every specialty equally, so rare specialties count as much as
        common ones.

        Loading the model prints a table of "unexpected" weights. That is
        normal: the pretraining heads of the checkpoint are dropped, and
        PyHealth adds a new classification head for our labels.
        """),
        code("""
        from pyhealth.datasets import get_dataloader, split_by_sample
        from pyhealth.models import TransformersModel
        from pyhealth.trainer import Trainer

        train_ds, val_ds, test_ds = split_by_sample(reports, [0.7, 0.1, 0.2], seed=0)
        model = TransformersModel(dataset=reports, model_name=MODEL_NAME)
        trainer = Trainer(model=model, metrics=["accuracy", "f1_macro", "f1_weighted"])
        trainer.train(
            train_dataloader=get_dataloader(train_ds, batch_size=16, shuffle=True),
            val_dataloader=get_dataloader(val_ds, batch_size=32),
            epochs=EPOCHS,
            monitor="f1_weighted",
            optimizer_params={"lr": 5e-5},
        )
        bert_scores = trainer.evaluate(get_dataloader(test_ds, batch_size=32))
        print(bert_scores)
        """),
        md("""
        ### A baseline to beat

        Bag-of-words models are fast and often strong on text classification.
        A TF-IDF + logistic regression baseline on the same split takes
        seconds; a language model should clearly beat it to justify its cost.
        """),
        code("""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        def texts_and_labels(ds):
            return [ds[i]["transcription"] for i in range(len(ds))], [int(ds[i]["medical_specialty"]) for i in range(len(ds))]

        x_train, y_train = texts_and_labels(train_ds)
        x_test, y_test = texts_and_labels(test_ds)
        tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000).fit(tfidf.fit_transform(x_train), y_train)
        pred = clf.predict(tfidf.transform(x_test))
        print(f"TF-IDF + LR   accuracy {accuracy_score(y_test, pred):.3f}  "
              f"F1 macro {f1_score(y_test, pred, average='macro'):.3f}  "
              f"F1 weighted {f1_score(y_test, pred, average='weighted'):.3f}")
        print(f"{MODEL_NAME.split('/')[-1]:13s} accuracy {bert_scores['accuracy']:.3f}  "
              f"F1 macro {bert_scores['f1_macro']:.3f}  F1 weighted {bert_scores['f1_weighted']:.3f}")
        """),
        md("""
        Scores on this dataset are modest for any model: several "specialties"
        (Surgery, Consult, Discharge Summary) describe the *kind* of document
        rather than the field, and overlap heavily with the others. Looking at
        the labels before modelling saves a lot of tuning.

        ## Part B. Medical coding from notes

        Medical coding assigns ICD codes to a hospital stay from its
        documentation. It is a **multilabel** problem: each note supports
        several codes at once. PyHealth's `MIMIC3ICD9Coding` task joins a
        patient's notes into one text and collects the diagnosis and
        procedure codes as labels:
        """),
        code("""
        from pyhealth.datasets import MIMIC3Dataset
        from pyhealth.tasks import MIMIC3ICD9Coding

        mimic = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "noteevents"],
            cache_dir="pyhealth_cache",
        )
        coding_task = MIMIC3ICD9Coding()
        print(coding_task.input_schema, "->", coding_task.output_schema)
        notes = mimic.set_task(coding_task)
        codes = notes.output_processors["icd_codes"].label_vocab
        print(f"{len(notes)} samples, {len(codes)} distinct codes")
        print("first sample has", int(notes[0]["icd_codes"].sum()), "codes")
        """),
        md("""
        With thousands of possible codes, most appearing only a few times,
        coding is hard. Two habits from practice:

        - Report **micro F1** (dominated by frequent codes) and **macro F1**
          (every code equal), plus ranking metrics such as precision@k.
        - Restrict to the most frequent codes (for example the top 50) for a
          first model, as most published MIMIC coding benchmarks do.

        The training code is identical to Part A; only the task changes. (On
        synthetic notes the text carries little information about the codes,
        so expect low scores; the workflow is what transfers to real MIMIC.)
        """),
        code("""
        from pyhealth.datasets import split_by_patient

        train_n, val_n, test_n = split_by_patient(notes, [0.7, 0.1, 0.2], seed=0)
        coder = TransformersModel(dataset=notes, model_name=MODEL_NAME)
        coding_trainer = Trainer(model=coder, metrics=["f1_micro", "f1_macro", "pr_auc_samples"])
        coding_trainer.train(
            train_dataloader=get_dataloader(train_n, batch_size=16, shuffle=True),
            val_dataloader=get_dataloader(val_n, batch_size=32),
            epochs=1,
            monitor="pr_auc_samples",
            optimizer_params={"lr": 5e-5},
        )
        print(coding_trainer.evaluate(get_dataloader(test_n, batch_size=32)))
        """),
        md("""
        ## Summary

        - A `text` field plus `TransformersModel(model_name=...)` fine-tunes
          any Hugging Face encoder; the task decides multiclass vs multilabel.
        - Pick the model size for your hardware; use a GPU runtime for
          full-size clinical models.
        - Always compare with a TF-IDF baseline, and inspect the labels before
          tuning.
        - Medical coding is multilabel and long-tailed: report micro and macro
          metrics, and start with frequent codes.
        """),
        *footer("08"),
    ]
