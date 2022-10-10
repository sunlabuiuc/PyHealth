from pyhealth.medcode.base_code import BaseCode


class CCSCM(BaseCode):
    VALID_MAPPINGS = ["ICD9CM", "ICD10CM"]

    def __init__(self, **kwargs):
        super(CCSCM, self).__init__(vocabulary="CCSCM", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = CCSCM()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["10"])
    print(code_sys.get_ancestors("10"))
    print(code_sys.map_to("10", "ICD9CM"))
