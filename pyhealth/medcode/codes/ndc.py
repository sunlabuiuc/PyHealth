from pyhealth.medcode.inner_map import InnerMap


# TODO: add standardize for different formats of NDC codes


class NDC(InnerMap):
    """National Drug Code."""

    def __init__(self, **kwargs):
        super(NDC, self).__init__(vocabulary="NDC", **kwargs)


if __name__ == "__main__":
    code_sys = NDC(refresh_cache=True)
    code_sys.stat()
    print("00527051210" in code_sys)
    print(code_sys.lookup("00527051210"))
    print(code_sys.get_ancestors("00527051210"))
    print(code_sys.get_descendants("00527051210"))
