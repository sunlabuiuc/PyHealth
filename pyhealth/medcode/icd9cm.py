from pyhealth.medcode.base_code import BaseCode
from pyhealth.medcode.utils import normalize_icd9cm


class ICD9CM(BaseCode):
    VALID_MAPPINGS = ["CCSCM"]

    def __init__(self, **kwargs):
        super(ICD9CM, self).__init__(vocabulary="ICD9CM",
                                     valid_mappings=self.VALID_MAPPINGS, **kwargs)

    def map_to(self, code, target):
        code = normalize_icd9cm(code)
        return super(ICD9CM, self).map_to(code, target)


if __name__ == "__main__":
    code_sys = ICD9CM()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["821.01"])
    print(code_sys.get_ancestors("821.01"))
    print(code_sys.map_to("821.01", "CCSCM"))
