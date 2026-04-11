from typing import Dict, List
from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

class HallucinationDetectionTask(BaseTask):
    """Task to evaluate the faithfulness of AI-generated clinical summaries.

    This task identifies hallucinations by comparing clinical notes (source) 
    against generated summaries. It produces binary labels where 1 indicates 
    a hallucination and 0 indicates a faithful summary.

    Attributes:
        input_schema: A dictionary mapping input feature names to their data types.
        output_schema: A dictionary mapping the target label name to its data type.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the task with predefined schemas for NLP processing.

        Args:
            **kwargs: Additional keyword arguments passed to the BaseTask parent class.
        """
        super(HallucinationDetectionTask, self).__init__(**kwargs)
        self.input_schema = {
            "source_text": str, 
            "summary_text": str
        }
        self.output_schema = {
            "label": int
        }

    def __call__(self, patient: Patient) -> List[Dict]:
        """Processes a patient's history into clinical summary faithfulness samples.

        This method iterates through hospital visits, aggregates professional 
        clinical notes, and pairs them with their corresponding AI-generated 
        summaries and expert-verified labels.

        Args:
            patient: A Patient object containing hospital visits and clinical 
                events backed by a Polars DataFrame.

        Returns:
            A list of dictionaries, where each dictionary represents one sample 
            containing 'visit_id', 'patient_id', 'source_text', 'summary_text', 
            and the binary 'label'.
        """
        samples = []

        # Get all visits for the patient using the polars partition
        visit_df = patient.get_events(event_type="visits", return_df=True)
        
        for visit_row in visit_df.to_dicts():
            v_id = visit_row.get("visits/visit_id")
            
            # Grab all notes linked to this specific hospital stay
            try: 
                notes = patient.get_events(
                    event_type="noteevents",
                    filters=[("visit_id", "==", v_id)],
                    return_df=False
                )

                # Combine all note text into a single context string
                source_context = " ".join(
                    [str(n.attr_dict.get("text", "")) for n in notes]
                ).strip()

                # Retrieve the summary and label from the visit attributes
                summary = visit_row.get("visits/ai_summary")
                label = visit_row.get("visits/hallucination_label")

                # Only add the sample if we have both the text and the target label
                if summary is not None and label is not None:
                    samples.append({
                        "visit_id": v_id,
                        "patient_id": patient.patient_id,
                        "source_text": source_context,
                        "summary_text": summary,
                        "label": int(label)
                    })

            except:
                pass

            

        return samples