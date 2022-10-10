from pyhealth.medcode import BaseCode


class ICD9CM(BaseCode):
    VALID_MAPPINGS = ["CCSCM"]

    def __init__(self):
        super(ICD9CM, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = ICD9CM()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['821.01'])
    print(nx.ancestors(code_sys.graph, '821.01'))
    print(code_sys.map_to("821.01", "CCSCM"))
