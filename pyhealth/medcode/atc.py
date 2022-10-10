from pyhealth.medcode import BaseCode


class ATC(BaseCode):
    VALID_MAPPINGS = ["NDC", "RxNorm"]

    def __init__(self):
        super(ATC, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = ATC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["N01AB07"])
    print(nx.ancestors(code_sys.graph, "N01AB07"))
    print(code_sys.map_to("N01AB07", "NDC"))
