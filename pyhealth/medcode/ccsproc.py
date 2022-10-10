from pyhealth.medcode import BaseCode


class CCSPROC(BaseCode):
    VALID_MAPPINGS = ["ICD9PROC", "ICD10PROC"]

    def __init__(self):
        super(CCSPROC, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = CCSPROC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['20'])
    print(nx.ancestors(code_sys.graph, '20'))
    print(code_sys.map_to("20", "ICD9PROC"))
