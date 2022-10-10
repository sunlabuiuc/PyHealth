from pyhealth.medcode import BaseCode


class RxNorm(BaseCode):
    VALID_MAPPINGS = ["NDC", "ATC"]

    def __init__(self, **kwargs):
        super(RxNorm, self).__init__(vocabulary="RxNorm", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = RxNorm()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["21914"])
    print(code_sys.get_ancestors("21914"))
    print(code_sys.map_to("21914", "NDC"))
    print(code_sys.map_to("21914", "ATC"))
