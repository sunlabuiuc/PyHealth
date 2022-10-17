from pyhealth.medcode.inner_map import InnerMap


# TODO: add postprocess

class ICD10PROC(InnerMap):

    def __init__(self, **kwargs):
        super(ICD10PROC, self).__init__(vocabulary="ICD10PROC", **kwargs)


if __name__ == "__main__":
    code_sys = ICD10PROC(refresh_cache=True)
    code_sys.stat()
    print("0LBG0ZZ" in code_sys)
    print(code_sys.lookup("0LBG0ZZ"))
    print(code_sys.get_ancestors("0LBG0ZZ"))
    print(code_sys.get_descendants("0LBG0"))
