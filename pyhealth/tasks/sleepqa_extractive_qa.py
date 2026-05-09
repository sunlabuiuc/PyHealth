from typing import Dict, List
from pyhealth.data import Event, Patient
from pyhealth.tasks.base_task import BaseTask

class SleepQAExtractiveQA(BaseTask):
    """Extractive Question Answering task for SleepQA.

    This task maps SleepQA events into samples containing a passage, 
    a question, and the answer span defined by character-level offsets.

    Input Schema:
        passage: raw text context.
        question: the sleep-related query.
    Output Schema:
        answer_text: the ground truth answer string.
        answer_start: char-level start index of the answer.
        answer_end: char-level end index of the answer (start + length).
    """
    task_name = "SleepQAExtractiveQA"
    
    # We define the schemas to guide the downstream DataLoader and Model
    input_schema = {
        "passage": "text", 
        "question": "text"
    }
    output_schema = {
        "answer_text": "text", 
        "answer_start": "integer",
        "answer_end": "integer"
    }

    def __call__(self, patient: Patient) -> List[Dict]:
        """Processes a patient object into QA samples.

        Args:
            patient: a Patient object containing SleepQA events.

        Returns:
            A list of sample dictionaries with span-position character offsets.
        """
        samples = []
        
        # Iterating through events specifically labeled for this dataset
        for event in patient.get_events(event_type="sleepqa"):
            
            # Extract raw values from the event attribute dictionary
            # Using bracket notation as requested
            passage = event["passage"]
            question = event["question"]
            answer_text = event["answer_text"]
            answer_start = int(event["answer_start"])
            
            # Calculate the character-level end offset
            # This is critical for span-based models to know the boundary
            answer_end = answer_start + len(answer_text)

            # Optional: Basic validation to ensure the offset matches the text
            # extracted_text = passage[answer_start:answer_end]
            # if extracted_text != answer_text:
            #     continue # Or handle mismatch logic here

            samples.append({
                "patient_id": patient.patient_id,
                "visit_id": event.visit_id,
                "passage": passage,
                "question": question,
                "answer_text": answer_text,
                "answer_start": answer_start,
                "answer_end": answer_end,
            })
            
        return samples

# Example Usage:
# task = SleepQAExtractiveQA()
# samples = task(patient_object)