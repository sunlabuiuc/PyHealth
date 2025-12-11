from types import SimpleNamespace

from pyhealth.tasks.cxr_report_generation import CXRReportGenerationMIMIC4


class DummyPatient:
    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type, start=None, end=None):
        return self._events.get(event_type, [])


def test_cxr_report_generation_basic():
    adm = SimpleNamespace(hadm_id="1")

    xray = SimpleNamespace(
        study_id="123",
        dicom_id="abc",
        ViewPosition="AP",
        image_path="/tmp/files/p10/p10000032/s123/abc.jpg",
    )

    note_txt = "FINDINGS: heart size normal.\nIMPRESSION: no acute disease."
    note = SimpleNamespace(study_id="123", radiology=note_txt)

    p = DummyPatient(
        patient_id="10000032",
        events={
            "admissions": [adm],
            "xrays_metadata": [xray],
            "radiology": [note],
        },
    )

    task = CXRReportGenerationMIMIC4(report_section="findings", view_positions=["AP"])
    out = task(p)

    assert len(out) == 1
    assert out[0]["study_id"] == "123"
    assert out[0]["dicom_id"] == "abc"
    assert "heart size normal" in out[0]["report"]


def test_cxr_report_generation_filters_empty():
    adm = SimpleNamespace(hadm_id="1")
    xray = SimpleNamespace(study_id="123", dicom_id="abc", ViewPosition="AP", image_path="/tmp/x.jpg")
    note = SimpleNamespace(study_id="123", radiology="IMPRESSION: ok.")  # no FINDINGS section

    p = DummyPatient(
        patient_id="10000032",
        events={"admissions": [adm], "xrays_metadata": [xray], "radiology": [note]},
    )

    task = CXRReportGenerationMIMIC4(report_section="findings", view_positions=["AP"], require_nonempty_report=True)
    out = task(p)
    assert out == []
