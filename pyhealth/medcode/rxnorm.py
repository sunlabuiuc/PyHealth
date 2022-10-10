from pyhealth.medcode import BaseCode


class RxNorm(BaseCode):
    VALID_MAPPINGS = ["NDC", "ATC"]

    def __init__(self):
        super(RxNorm, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = RxNorm()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["21914"])
    print(nx.ancestors(code_sys.graph, "21914"))
    print(code_sys.map_to("21914", "NDC"))
    print(code_sys.map_to("21914", "ATC"))
