from pyhealth.medcode.inner_map import InnerMap


class CCSPROC(InnerMap):
    """Classification of Diseases, Procedure."""

    def __init__(self, **kwargs):
        super(CCSPROC, self).__init__(vocabulary="CCSPROC", **kwargs)


if __name__ == "__main__":
    code_sys = CCSPROC(refresh_cache=True)
    code_sys.stat()
    print("1" in code_sys)
    print(code_sys.lookup("20"))
    print(code_sys.get_ancestors("20"))
    print(code_sys.get_descendants("20"))
