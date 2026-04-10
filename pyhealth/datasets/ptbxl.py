import pandas as pd
from pathlib import Path

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

"""Full list of possible diagnoses for the PTB-XL dataset provided here: https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv

Not all codes are present in the data but they are included for completeness, as referenced in the Data Description section here: https://physionet.org/content/challenge-2020/1.0.2/
"""
SNOMED_CT_ABBREVIATION = {
    "270492004":	    IAVB,
    "195042002":	    IIAVB,
    "164951009":	    abQRS,
    "426664006":	    AJR,
    "57054005":    	    AMI,
    "413444003":        AMIs,
    "426434006":	    AnMIs,
    "54329005":    	    AnMI,
    "251173003":	    AB,
    "164889003":	    AF,
    "195080001":	    AFAFL,
    "164890007":	    AFL,
    "195126007":	    AH,
    "251268003":	    AP,
	"713422000":	    ATach,
    "29320008":        	AVJR,
	"233917008":	    AVB,
    "251170000":	    BPAC,
    "74615001":	        BTS,
    "426627000":	    Brady,
    "6374002":    	    BBB,
    "698247007":	    CD,    
    "426749004":	    CAF,
	"413844008":	    CMI,
    "27885002":	        CHB,
    "713427006":	    CRBBB,
    "204384007":	    CIAHB,
    "53741008":	        CHD,
    "77867006":    	    SQT,
    "82226007":    	    DIB,
    "428417006":	    ERe,
    "13640000":    	    FB,
    "84114007":    	    HF,
    "368009":    	    HVD,
    "251259000":	    HTV,
    "49260003":        	IR,
    "251120003":	    ILBBB,
    "713426002":	    IRBBB,
    "251200008":	    ICA,
    "425419005":	    IIs,
    "704997005":	    ISTD,
    "426995002":	    JE,
    "251164006":	    JPC,
    "426648003":	    JTach,
    "425623009":	    LIs,
    "445118002":	    LAnFB,
    "253352002":	    LAA,
    "67741000119109":	LAE,
    "446813000":	    LAH,
    "39732003":        	LAD,
	"164909002":	    LBBB,
    "445211001":	    LPFB,
    "164873001":	    LVH,
	"370365005":	    LVS,
    "251146004":	    LQRSV,
    "54016002":        	MoI,
    "164865005":	    MI,
    "164861001":	    MIs,
    "698252002":	    NSIVCB,
	"428750005":	    NSSTTA,
	"164867002":	    OldMI,
	"10370003":	        PR,
	"251182009":	    VPVC,
	"282825002":	    PAF,
	"67198005":	        PSVT,
	"425856008":	    PVT,
	"284470004":	    PAC,
	"427172004":	    PVC,
	"17338001":    	    VPB,
	"164947007":        LPR,
	"111975006":	    LQT,
	"164917005":	    QAb,
	"164921003":	    RAb,
	"314208002":	    RAF,
	"253339007":	    RAAb,
	"446358003":	    RAH,
	"47665007":        	RAD,
	"59118001":	        RBBB,
	"89792004":	        RVH,
	"55930002":        	STC,
	"49578007":	        SPRI,
	"65778007":    	    SAB,
	"427393009":	    SA,
	"426177001":	    SB,
	"60423000":	        SND,
	"426783006":	    NSR,
	"427084000":	    STach,
	"429622005":	    STD,
	"164931005":	    STE,
	"164930006":	    STIAb,
	"251168009":	    SVB,
	"63593006":	        SVPB,
	"426761007":	    SVT,
	"251139008":	    ALR,
	"164934002":	    TAb,
	"59931005":	        TInv,
	"266257000":	    TIA,
	"164937009":	    UAb,
	"11157007":	        VBig,
	"164884008":	    VEB,
	"75532003":	        VEsB,
	"81898007":	        VEsR,
	"164896001":	    VF,
	"111288001":	    VFL,
	"266249003":	    VH,
	"251266004":	    VPP,
	"195060002":	    VPEx,
	"164895002":	    VTach,
	"251180001":	    VTrig,
	"195101003":	    WAP,
	"74390002":	        WPW
}

class PTBXLDataset(BaseDataset):
    """Base dataset for the PTB-XL ECG dataset

    PTB-XL is a publically available electrocardiography dataset. Contains 21837 samples from 18885 patients, all approximately 10 seconds in duration.

    Dataset is available here: https://www.kaggle.com/datasets/physionet/ptbxl-electrocardiography-database
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            tables=["ptbxl"],
            dataset_name="ptbxl",
            config_path=None,
            **kwargs,
        ) 

    def load_data(self) -> dd.DataFrame:
        """Returns a dataframe with each individual row corresponding to each .hea/.mat file combination in the PTB-XL dataset.

        Returns:
          dd.DataFrame: Dataframe with one row per record
        """
        root_path = Path(self.root)
        files = root_path.glob("*.hea")

        if not files:
            raise FileNotFoundError(f"No .hea files found in {self.root}. Are you sure you have the right directory?")

        # Should all be in the same order to make the training / validation / test splits deterministic 
        files = sorted(files)
        
        logger.info(f"Found {len(files)} .hea files")

        rows = []
        for hea_file in files:
            age = None
            sex = None
            dx = []
            
            with open(hea_file, "r") as f:
                for line in f:
                    line = line.strip()

                    # Parse individual lines
                    if line.startswith("#Age:"):
                        try:
                            age = int(line.split(":")[1].strip())
                        except ValueError:
                            age = None
                    elif line.startswith("#Sex:"):
                        sex = line.split(":")[1].strip()
                    elif line.startswith("#Dx:"):
                        dx = [x.strip() for x in line.split(":")[1].split(",")]

            # Map diagnosis codes to the abbreviations (may need them for tasks)
            dx_abbreviations = [SNOMED_CT_ABBREVIATION[x] for x in dx if x in SNOMED_CT_ABBREVIATION]

            # Append required data to the list                 
            rows.append({
                  "patient_id":     hea_file.stem,
                  "event_type":     "ptbxl",
                  "timestamp":      pd.NaT,
                  "ptbxl/mat":      str(root_path / f"{hea_file.stem}.mat"),
                  "ptbxl/age":      age,
                  "ptbxl/sex":      sex,
                  "ptbxl/dx_codes": ".".join(dx)
                  "ptbxl/dx_abbreviations": ","./join(dx_abbreviations)
            })

        df = pd.Dataframe(rows)
        # TBD - Will need train and test splits

        logger.info(f"Parsed {len(df)} records.")
        return dd.from_pandas(df, npartitions=1)
        
      
        
