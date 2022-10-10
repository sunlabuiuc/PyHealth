from pyhealth.medcode import BaseCode


class NDC(BaseCode):
    VALID_MAPPINGS = ["RxNorm", "ATC"]

    def __init__(self):
        super(NDC, self).__init__(self.VALID_MAPPINGS)


if __name__ == "__main__":
    import networkx as nx

    code_sys = NDC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["00527051210"])
    print(nx.ancestors(code_sys.graph, "00527051210"))
    print(code_sys.map_to("00527051210", "RxNorm"))
    print(code_sys.map_to("00527051210", "ATC"))
