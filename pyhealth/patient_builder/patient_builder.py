class PatientBuilder:
    """
    Cohort builder returning PyHealth Patient objects.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.patient_ids = list(
            dataset.unique_patient_ids
        )

    def with_condition(self, condition_code):
        """
        Filter patients having the specified condition.
        """

        filtered_ids = []

        for patient_id in self.patient_ids:

            patient = self.dataset.get_patient(
                patient_id
            )

            conditions = patient.get_events(
                event_type="condition_occurrence"
            )

            for condition in conditions:

                code = self.dataset.get_condition_code(
                    condition
                )

                if code == str(condition_code):
                    filtered_ids.append(
                        patient_id
                    )
                    break

        self.patient_ids = filtered_ids

        return self

    def with_min_visits(self, min_visits):
        """
        Filter patients with at least min_visits.
        """

        filtered_ids = []

        for patient_id in self.patient_ids:

            patient = self.dataset.get_patient(
                patient_id
            )

            visit_count = (
                self.dataset.get_visit_count(
                    patient
                )
            )

            if visit_count >= min_visits:
                filtered_ids.append(
                    patient_id
                )

        self.patient_ids = filtered_ids

        return self

    def build(self):
        """
        Return PyHealth Patient objects.
        """

        return [
            self.dataset.get_patient(pid)
            for pid in self.patient_ids
        ]