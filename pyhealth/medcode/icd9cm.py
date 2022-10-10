from pyhealth.medcode import BaseCode


class ICD9CM(BaseCode):
    VALID_MAPPINGS = ["CCSCM"]

    def __init__(self, **kwargs):
        super(ICD9CM, self).__init__(vocabulary="ICD9CM", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = ICD9CM()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['821.01'])
    print(code_sys.get_ancestors('821.01'))
    print(code_sys.map_to("821.01", "CCSCM"))
