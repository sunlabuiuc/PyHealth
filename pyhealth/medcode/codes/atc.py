from pyhealth.medcode.inner_map import InnerMap


class ATC(InnerMap):

    def __init__(self, **kwargs):
        super(ATC, self).__init__(vocabulary="ATC", **kwargs)


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
