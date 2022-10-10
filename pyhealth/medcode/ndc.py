from pyhealth.medcode.base_code import BaseCode


class NDC(BaseCode):
    VALID_MAPPINGS = ["RxNorm", "ATC"]

    def __init__(self, **kwargs):
        super(NDC, self).__init__(vocabulary="NDC", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = NDC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["00527051210"])
    print(code_sys.get_ancestors("00527051210"))
    print(code_sys.map_to("00527051210", "RxNorm"))
    print(code_sys.map_to("00527051210", "ATC"))
