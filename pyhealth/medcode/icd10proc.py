from pyhealth.medcode.base_code import BaseCode


class ICD10PROC(BaseCode):
    VALID_MAPPINGS = ["CCSPROC"]

    def __init__(self, **kwargs):
        super(ICD10PROC, self).__init__(vocabulary="ICD10PROC", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = ICD10PROC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["0LBG0ZZ"])
    print(code_sys.get_ancestors("0LBG0ZZ"))
    print(code_sys.map_to("0LBG0ZZ", "CCSPROC"))
