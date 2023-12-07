from pyhealth.medcode.inner_map import InnerMap


class UMLS(InnerMap):
    """UMLS."""

    def __init__(self, **kwargs):
        super(UMLS, self).__init__(vocabulary="UMLS", **kwargs)


if __name__ == "__main__":
    code_sys = UMLS(refresh_cache=True)
    code_sys.stat()
    print(code_sys.lookup("C0000768"))
