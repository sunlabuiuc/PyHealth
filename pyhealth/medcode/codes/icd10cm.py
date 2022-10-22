from pyhealth.medcode.inner_map import InnerMap


# TODO: add postprocess

class ICD10CM(InnerMap):

    def __init__(self, **kwargs):
        super(ICD10CM, self).__init__(vocabulary="ICD10CM", **kwargs)

    def standardize(self, code: str):
        """Standardizes ICD10CM code."""
        if "." in code:
            return code
        if len(code) <= 3:
            return code
        return code[:3] + "." + code[3:]


if __name__ == "__main__":
    code_sys = ICD10CM(refresh_cache=True)
    code_sys.stat()
    print("A00.0" in code_sys)
    print(code_sys.lookup("D50.0"))
    print(code_sys.get_ancestors("D50.0"))
    print(code_sys.get_descendants("D50"))
