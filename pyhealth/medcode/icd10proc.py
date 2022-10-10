from pyhealth.medcode import BaseCode


class ICD10PROC(BaseCode):
    VALID_MAPPINGS = ["CCSPROC"]

    def __init__(self):
        super(ICD10PROC, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = ICD10PROC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['0LBG0ZZ'])
    print(nx.ancestors(code_sys.graph, '0LBG0ZZ'))
    print(code_sys.map_to("0LBG0ZZ", "CCSPROC"))
