from pyhealth.medcode.base_code import BaseCode


class ATC(BaseCode):
    VALID_MAPPINGS = ["NDC", "RxNorm"]

    def __init__(self, **kwargs):
        super(ATC, self).__init__(vocabulary="ATC", valid_mappings=self.VALID_MAPPINGS, **kwargs)


if __name__ == "__main__":
    code_sys = ATC()
    print(len(code_sys.graph.nodes))
    print(len(code_sys.graph.edges))
    print(code_sys.graph.nodes["N01AB07"])
    print(code_sys.get_ancestors("N01AB07"))
    print(code_sys.map_to("N01AB07", "NDC"))
