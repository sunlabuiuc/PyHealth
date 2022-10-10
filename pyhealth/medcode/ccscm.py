from pyhealth.medcode import BaseCode


class CCSCM(BaseCode):
    VALID_MAPPINGS = ["ICD9CM", "ICD10CM"]

    def __init__(self):
        super(CCSCM, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = CCSCM()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes['10'])
    print(nx.ancestors(code_sys.graph, '10'))
    print(code_sys.map_to("10", "ICD9CM"))
