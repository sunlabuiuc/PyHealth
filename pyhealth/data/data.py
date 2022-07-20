class Visit:
    """ Contains information about a single visit """

    def __init__(self,
                 visit_id,
                 patient_id,
                 conditions,
                 procedures,
                 drugs,
                 visit_info=None):
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.conditions = conditions
        self.procedures = procedures
        self.drugs = drugs
        self.visit_info = visit_info


class Patient:
    """ Contains information about a single patient """

    def __init__(self,
                 patient_id,
                 visits,
                 patient_info=None):
        self.patient_id = patient_id
        self.visits = visits
        self.patient_info = patient_info
