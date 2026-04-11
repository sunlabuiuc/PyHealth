from pyhealth.medcode.inner_map import InnerMap


# TODO: add convert


class ICD9PROC(InnerMap):
    """9-th International Classification of Diseases, Procedure."""

    def __init__(self, **kwargs):
        super(ICD9PROC, self).__init__(vocabulary="ICD9PROC", **kwargs)

    @staticmethod
    def standardize(code: str):
        """Standardizes ICD9PROC code."""
        if "." in code:
            return code
        if len(code) <= 2:
            return code
        return code[:2] + "." + code[2:]


if __name__ == "__main__":
    code_sys = ICD9PROC(refresh_cache=True)
    code_sys.stat()
    print("81.01" in code_sys)
    print(code_sys.lookup("01.31"))
    print(code_sys.get_ancestors("01.31"))
    print(code_sys.get_descendants("01"))
