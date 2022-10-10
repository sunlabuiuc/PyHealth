from pyhealth.medcode.base_code import BaseCode


class ICD10CM(BaseCode):
    VALID_MAPPINGS = ["CCSCM"]

    def __init__(self, **kwargs):
        super(ICD10CM, self).__init__(vocabulary="ICD10CM", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = ICD10CM()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["D50.0"])
    print(code_sys.get_ancestors("D50.0"))
    print(code_sys.map_to("D50.0", "CCSCM"))
