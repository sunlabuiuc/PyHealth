from pyhealth.medcode.inner_map import InnerMap


# TODO: add DDI

class ATC(InnerMap):

    def __init__(self, **kwargs):
        super(ATC, self).__init__(vocabulary="ATC", **kwargs)

    def postprocess(self, code: str, level=5):
        if type(level) is str:
            level = int(level)
        assert level in [1, 2, 3, 4, 5]
        if level == 1:
            return code[:1]
        elif level == 2:
            return code[:3]
        elif level == 3:
            return code[:4]
        elif level == 4:
            return code[:5]
        else:
            return code


if __name__ == "__main__":
    code_sys = ATC(refresh_cache=True)
    code_sys.stat()
    print(code_sys.lookup("N01AB07"))
    print(code_sys.lookup("N01AB07", attribute="level"))
    print(code_sys.lookup("N01AB07", attribute="description"))
    print(code_sys.lookup("N01AB07", attribute="indication"))
    print(code_sys.lookup("N01AB07", attribute="smiles"))
    print(code_sys.lookup("N01AB07", attribute="drugbank_id"))
    print(code_sys.get_ancestors("N01AB07"))
    print(code_sys.get_descendants("N01AB"))
