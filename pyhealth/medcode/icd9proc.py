from pyhealth.medcode import BaseCode


class ICD9PROC(BaseCode):
    VALID_MAPPINGS = ["CCSPROC"]

    def __init__(self, **kwargs):
        super(ICD9PROC, self).__init__(vocabulary="ICD9PROC", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = ICD9PROC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['01.31'])
    print(code_sys.get_ancestors('01.31'))
    print(code_sys.map_to("01.31", "CCSPROC"))
