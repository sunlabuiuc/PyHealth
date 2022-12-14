from pyhealth.medcode.inner_map import InnerMap


# TODO: add convert


class ICD9CM(InnerMap):
    """9-th International Classification of Diseases, Clinical Modification."""

    def __init__(self, **kwargs):
        super(ICD9CM, self).__init__(vocabulary="ICD9CM", **kwargs)

    @staticmethod
    def standardize(code: str):
        """Standardizes ICD9CM code."""
        if "." in code:
            return code
        if code.startswith("E"):
            if len(code) <= 4:
                return code
            return code[:4] + "." + code[4:]
        else:
            if len(code) <= 3:
                return code
            return code[:3] + "." + code[3:]


if __name__ == "__main__":
    code_sys = ICD9CM(refresh_cache=True)
    code_sys.stat()
    print("821.01" in code_sys)
    print(code_sys.lookup("82101"))
    print(code_sys.get_ancestors("821.01"))
    print(code_sys.get_descendants("821"))
