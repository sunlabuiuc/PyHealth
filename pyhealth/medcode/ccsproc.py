from pyhealth.medcode.base_code import BaseCode


class CCSPROC(BaseCode):
    VALID_MAPPINGS = ["ICD9PROC", "ICD10PROC"]

    def __init__(self, **kwargs):
        super(CCSPROC, self).__init__(vocabulary="CCSPROC", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = CCSPROC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["20"])
    print(code_sys.get_ancestors("20"))
    print(code_sys.map_to("20", "ICD9PROC"))
