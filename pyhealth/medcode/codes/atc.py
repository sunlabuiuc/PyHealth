from typing import List

from pyhealth.medcode.inner_map import InnerMap
from pyhealth.medcode.utils import download_and_read_csv


class ATC(InnerMap):
    """Anatomical Therapeutic Chemical."""

    def __init__(self, **kwargs):
        super(ATC, self).__init__(vocabulary="ATC", **kwargs)
        self.ddi = dict()

    @staticmethod
    def convert(code: str, level=5):
        """Convert ATC code to a specific level."""
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

    def get_ddi(
        self, gamenet_ddi: bool = False, refresh_cache: bool = False
    ) -> List[str]:
        """Gets the drug-drug interactions (DDI).

        Args:
            gamenet_ddi: Whether to use the DDI from the GAMENet paper,
                which is a subset of the DDI from the ATC.
            refresh_cache: Whether to refresh the cache. Default is False.
        """
        filename = "DDI_GAMENet.csv" if gamenet_ddi else "DDI.csv"
        if filename not in self.ddi or refresh_cache:
            df = download_and_read_csv(filename, refresh_cache)
            ddi = []
            for idx, row in df.iterrows():
                ddi.append([row["ATC i"], row["ATC j"]])
            self.ddi[filename] = ddi
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
    print(len(code_sys.get_ddi(gamenet_ddi=True)))
    print(len(code_sys.get_ddi(gamenet_ddi=False)))
