from pyhealth.medcode.inner_map import InnerMap
from pyhealth.medcode.utils import download_and_read_csv


class ATC(InnerMap):

    def __init__(self, **kwargs):
        super(ATC, self).__init__(vocabulary="ATC", **kwargs)
        self.ddi = dict()

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

    @staticmethod
    def load_ddi(filename: str, refresh_cache: bool = False):
        df = download_and_read_csv(filename, refresh_cache)
        ddi = []
        for idx, row in df.iterrows():
            ddi.append([row["ATC 1"], row["ATC 2"]])
        return ddi

    def get_ddi(self, top_40: bool = True, refresh_cache: bool = False):
        filename = "DDI_TOP40.csv" if top_40 else "DDI_ALL.csv"
        if filename not in self.ddi or refresh_cache:
            self.ddi[filename] = self.load_ddi(filename, refresh_cache=refresh_cache)
        return self.ddi[filename]


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
    print(len(code_sys.get_ddi(top_40=True)))
    print(len(code_sys.get_ddi(top_40=False)))
