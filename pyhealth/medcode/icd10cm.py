from pyhealth.medcode import BaseCode


class ICD10CM(BaseCode):
    VALID_MAPPINGS = ["CCSCM"]

    def __init__(self):
        super(ICD10CM, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = ICD10CM()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['D50.0'])
    print(nx.ancestors(code_sys.graph, 'D50.0'))
    print(code_sys.map_to("D50.0", "CCSCM"))
