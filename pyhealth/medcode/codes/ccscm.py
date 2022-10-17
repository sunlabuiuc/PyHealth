from pyhealth.medcode.inner_map import InnerMap


# TODO: code names are abbreviated

class CCSCM(InnerMap):

    def __init__(self, **kwargs):
        super(CCSCM, self).__init__(vocabulary="CCSCM", **kwargs)


if __name__ == "__main__":
    code_sys = CCSCM(refresh_cache=True)
    code_sys.stat()
    print("20" in code_sys)
    print(code_sys.lookup("10"))
    print(code_sys.get_ancestors("10"))
    print(code_sys.get_descendants("10"))
