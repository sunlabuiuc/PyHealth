from pyhealth.medcode.inner_map import InnerMap


class RxNorm(InnerMap):
    """RxNorm."""

    def __init__(self, **kwargs):
        super(RxNorm, self).__init__(vocabulary="RxNorm", **kwargs)


if __name__ == "__main__":
    code_sys = RxNorm(refresh_cache=True)
    code_sys.stat()
    print("21914" in code_sys)
    print(code_sys.graph.nodes["21914"])
    print(code_sys.get_ancestors("21914"))
    print(code_sys.get_descendants("21914"))
