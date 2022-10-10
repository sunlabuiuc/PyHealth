from pyhealth.medcode import BaseCode


class ICD9PROC(BaseCode):
    VALID_MAPPINGS = ["CCSPROC"]

    def __init__(self):
        super(ICD9PROC, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = ICD9PROC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['01.31'])
    print(nx.ancestors(code_sys.graph, '01.31'))
    print(code_sys.map_to("01.31", "CCSPROC"))
