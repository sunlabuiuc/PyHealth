# TODO

class OMOPBaseDataset:
    """
    MIMIC-III data object
        - the original MIMIC-III medication is encoded by RxNorm (the column name uses "NDC")
        - when initialize, input the target_code and the according code_map
    For example,
        - target_code = 'ATC4'
        - code_map = {RxNorm: ATC4} mapping dict
    """

    def __init__(self, table_names=['med', 'diag', 'prod']):
        # table_names
        self.table_names = table_names
        # {table_name: df}
        self.tables = {}
        # {visit_id: {table_name: df}}
        self.visit_dict = {}
        # df with visit_id as index
        self.pat_to_visit = {}
        # {table_name: CodetoIndex Object}
        self.maps = {}
        # encoded {visit_id: {table_name: df}}
        self.encoded_visit_dict = {}
        # dataloader
        self.train_loader = None
        self.test_loader = None
        self.vocab_size = None
        self.ddi_adj = None

        self._get_data_tables()
        self._get_pat_and_visit_dicts()

        # map med coding to ATC3
        # First generate RxNorm (the raw drug coding) to ATC4, then use [:-1] to get ATC3
        target_code = 'ATC4'
        self.tool = CodeMapping('RxNorm', target_code)
        self.tool.load()
        self._encode_visit_info(self.tool.RxNorm_to_ATC4)
        self._generate_ddi_adj()
        # self._summarize()

    def _get_data_tables(self):
        """
        INPUT:
            - self.table_names <string list>: for example, ["med", "diag", "prod"]
        OUTPUT:
            - self.tables <dict>: key is the table name, value is the dataframe for each table
        """
        import psycopg2 as psql
        conn = psql.connect(
            database="postgres",
            user='postgres',
            password='chaoqi',
            host='127.0.0.1',
            port='5432',
        )
        cursor = conn.cursor()

        def fetch_diagnosis():
            cursor.execute("set search_path to omop;")
            cursor.execute(
                """
                SELECT person_id, visit_occurrence_id, condition_concept_id FROM condition_occurrence
                """
            )
            data = cursor.fetchall()
            diag = pd.DataFrame(data, columns=['person_id', 'visit_occurrence_id', 'condition_concept_id'])
            diag.fillna(method='pad', inplace=True)
            diag.drop_duplicates(inplace=True)
            return diag

        def fetch_procedure():
            cursor.execute("set search_path to omop;")
            cursor.execute(
                """
                SELECT person_id, visit_occurrence_id, procedure_concept_id FROM procedure_occurrence
                """
            )
            data = cursor.fetchall()
            prod = pd.DataFrame(data, columns=['person_id', 'visit_occurrence_id', 'procedure_concept_id'])
            prod.fillna(method='pad', inplace=True)
            prod.drop_duplicates(inplace=True)
            return prod

        def fetch_medication():
            cursor.execute("set search_path to mimiciii;")
            cursor.execute(
                """
                SELECT x.mimic_id, y.mimic_id, x.NDC
                FROM (
                    SELECT b.mimic_id, a.hadm_id, a.ndc 
                    FROM prescriptions as a 
                    LEFT JOIN patients as b 
                    ON a.subject_id = b.subject_id
                ) as x
                LEFT JOIN admissions as y
                ON x.hadm_id = y.hadm_id
                """
            )
            data = cursor.fetchall()
            med = pd.DataFrame(data, columns=['person_id', 'visit_occurrence_id', 'drug_concept_id'])
            med.fillna(method='pad', inplace=True)
            med.drop_duplicates(inplace=True)
            return med

        self.tables['med'] = fetch_medication()
        self.tables['diag'] = fetch_diagnosis()
        self.tables['prod'] = fetch_procedure()

    def _get_pat_and_visit_dicts(self):
        """
        INPUT:
            - self.tables <dict>: key is the table name, value is the dataframe for each table
        OUTPUT:
            - self.pat_to_visit <dict>: key is the pat id, value is a list of visits
            - self.visit_dict <dict>: key is the visit id, value is a dict of <table_name: df>
        """
        for name, df in self.tables.items():
            for pat_id, pat_info in tqdm(df.groupby('person_id')):
                self.pat_to_visit[pat_id] = []
                for HAMD_id, HADM_info in pat_info.groupby('visit_occurrence_id'):
                    self.pat_to_visit[pat_id].append(HAMD_id)
                    if HAMD_id not in self.visit_dict:
                        self.visit_dict[HAMD_id] = {}
                    self.visit_dict[HAMD_id][name] = HADM_info
        print("generated .pat_to_visit!")
        print("generated .visit_dict!")

    def _encode_visit_info(self, code_map):

        # initialize code-to-index map
        med_map = CodeIndexing()
        diag_map = CodeIndexing()
        prod_map = CodeIndexing()

        def get_atc3(x):
            # one rxnorm maps to one or more ATC3
            result = []
            for rxnorm in x:
                if rxnorm in code_map:
                    result += code_map[rxnorm]
            result = np.unique([item[:-1] for item in result]).tolist()
            return result

        encoded_visit_dict = {}
        for visit_id, value in tqdm(self.visit_dict.items()):

            # if one of them does not exist, then drop this visit
            if 'med' not in value or 'diag' not in value or 'prod' not in value: continue
            cur_med, cur_diag, cur_prod = value['med'], value['diag'], value['prod']

            # RxNorm->ATC3 coded med
            cur_med = get_atc3(["{:011}".format(med) for med in cur_med.drug_concept_id.unique().astype('int')])
            # ICD9 coded diag
            cur_diag = cur_diag.condition_concept_id.unique()
            # ICD9 coded prod
            cur_prod = cur_prod.procedure_concept_id.unique()

            # if one of them does not exist, then drop this visit
            if len(cur_med) * len(cur_diag) * len(cur_prod) == 0: continue

            # build the maps
            med_map.build(cur_med)
            diag_map.build(cur_diag)
            prod_map.build(cur_prod)

            encoded_visit_dict[visit_id] = {}
            encoded_visit_dict[visit_id]['med'] = med_map.encodes(cur_med)
            encoded_visit_dict[visit_id]['diag'] = diag_map.encodes(cur_diag)
            encoded_visit_dict[visit_id]['prod'] = prod_map.encodes(cur_prod)

        self.encoded_visit_dict = encoded_visit_dict
        print("generated .encoded_visit_dict!")

        self.maps = {
            'med': med_map,
            'diag': diag_map,
            'prod': prod_map,
        }
        self.voc_size = (len(diag_map), len(prod_map), len(med_map))

        print("generated .maps (for code to index mappings)!")

    # get ddi matrix based on ATC3 coding
    def _generate_ddi_adj(self):
        cid2atc_dic = defaultdict(set)
        med_voc_size = self.voc_size[2]

        if not os.path.exists('./data/drug-atc.csv'):
            cid_to_ATC6 = request.urlopen(
                "https://drive.google.com/uc?id=1CVfa91nDu3S_NTxnn5GT93o-UfZGyewI").readlines()
            with open('./data/drug-atc.csv', 'w') as outfile:
                for line in cid_to_ATC6:
                    print(str(line[:-1]), file=outfile)
        else:
            cid_to_ATC6 = open('./data/drug-atc.csv', 'r').readlines()

        for line in cid_to_ATC6:
            line_ls = str(line[:-1]).split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if atc[:4] in self.maps['med'].code_to_idx:
                    cid2atc_dic[cid[2:]].add(atc[:4])

        # load ddi_df
        print('load severe ddi pairs from https://drive.google.com/uc?id=1R88OIhn-DbOYmtmVYICmjBSOIsEljJMh!')
        print('ddi info is from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing!')
        if not os.path.exists('./data/drug-DDI-TOP40.csv'):
            ddi_df = pd.read_csv(request.urlopen('https://drive.google.com/uc?id=1R88OIhn-DbOYmtmVYICmjBSOIsEljJMh'))
            ddi_df.to_csv('./data/drug-DDI-TOP40.csv', index=False)
        else:
            ddi_df = pd.read_csv('./data/drug-DDI-TOP40.csv')

        # ddi adj
        ddi_adj = np.zeros((med_voc_size, med_voc_size))
        for index, row in ddi_df.iterrows():
            # ddi
            cid1 = row['STITCH 1']
            cid2 = row['STITCH 2']

            # cid -> atc_level3
            for atc_i in cid2atc_dic[cid1]:
                for atc_j in cid2atc_dic[cid2]:
                    ddi_adj[self.maps['med'].encode(atc_i), self.maps['med'].encode(atc_j)] = 1
                    ddi_adj[self.maps['med'].encode(atc_j), self.maps['med'].encode(atc_i)] = 1

        self.ddi_adj = ddi_adj

    def _generate_ehr_adj_for_GAMENet(self, data_train):
        """
        generate the ehr graph adj for GAMENet model input
        - loop over the training data to check whether any med pair appear
        """
        ehr_adj = np.zeros((self.voc_size[2], self.voc_size[2]))
        for patient in data_train:
            for visit in patient:
                for idx1, med1 in enumerate(visit[2]):
                    for idx2, med2 in enumerate(visit[2]):
                        if idx1 >= idx2: continue
                        ehr_adj[med1, med2] = 1
                        ehr_adj[med2, med1] = 1
        return ehr_adj

    def _generate_ddi_mask_H_for_SafeDrug(self, ATC4_to_SMILES):
        # idx_to_SMILES
        SMILES = [[] for _ in range(self.voc_size[2])]
        # each idx contains what segments
        fraction = [[] for _ in range(self.voc_size[2])]

        for atc4, smiles_ls in ATC4_to_SMILES.items():
            if atc4[:-1] in self.maps['med'].code_to_idx:
                pos = self.maps['med'].encode(atc4[:-1])
                SMILES[pos] += smiles_ls
                for smiles in smiles_ls:
                    if smiles != 'nan':
                        try:
                            m = Chem.BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles))
                            for frac in m:
                                fraction[pos].add(frac)
                        except:
                            pass
        # all segment set
        fraction_set = []
        for i in fraction:
            fraction_set += i
        fraction_set = list(set(fraction_set))  # set of all segments

        # ddi_mask
        ddi_mask_H = np.zeros((self.voc_size[2], len(fraction_set)))
        for idx, cur_fraction in enumerate(fraction):
            for frac in cur_fraction:
                ddi_mask_H[idx, fraction_set.index(frac)] = 1
        return ddi_mask_H, SMILES

    def _generate_med_molecule_info_for_SafeDrug(self, SMILES, radius=1):

        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        MPNNSet, average_index = [], []

        for smilesList in SMILES:
            """Create each data with the above defined functions."""
            counter = 0  # counter how many drugs are under that ATC-3
            for smiles in smilesList:
                try:
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    atoms = create_atoms(mol, atom_dict)
                    molecular_size = len(atoms)
                    i_jbond_dict = create_ijbonddict(mol, bond_dict)
                    fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                        fingerprint_dict, edge_dict)
                    adjacency = Chem.GetAdjacencyMatrix(mol)
                    # if fingerprints.shape[0] == adjacency.shape[0]:
                    for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                        fingerprints = np.append(fingerprints, 1)

                    fingerprints = torch.LongTensor(fingerprints)
                    adjacency = torch.FloatTensor(adjacency)
                    MPNNSet.append((fingerprints, adjacency, molecular_size))
                    counter += 1
                except:
                    continue

            average_index.append(counter)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """

        N_fingerprint = len(fingerprint_dict)
        # transform into projection matrix
        n_col = sum(average_index)
        n_row = len(average_index)

        average_projection = np.zeros((n_row, n_col))
        col_counter = 0
        for i, item in enumerate(average_index):
            if item > 0:
                average_projection[i, col_counter: col_counter + item] = 1 / item
            col_counter += item

        return [MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)]

    def get_dataloader(self, MODEL):
        """
        get the dataloaders for MODEL, since different models has different data loader (input formats are different)
            - data <list>: each element is a patient record
                - data[0] <list>: each element is a visit
                    - data[0][0] <list>: diag encoded list for this visit
                    - data[0][1] <list>: prod encoded list for this visit
                    - data[0][2] <list>: med encoded list for this visit
        """
        data = []
        for pat_id, visit_ls in self.pat_to_visit.items():
            visit_ls = sorted(visit_ls)
            cur_pat = []
            for visit_id in visit_ls:
                if visit_id not in self.encoded_visit_dict: continue
                value = self.encoded_visit_dict[visit_id]
                cur_med, cur_diag, cur_prod = value['med'], value['diag'], value['prod']
                cur_diag_info = list(map(int, cur_diag.split(',')))
                cur_prod_info = list(map(int, cur_prod.split(',')))
                cur_med_info = list(map(int, cur_med.split(',')))
                cur_pat.append([cur_diag_info, cur_prod_info, cur_med_info, pat_id, visit_id])
            if len(cur_pat) <= 1: continue
            data.append(cur_pat)

        # data split
        split_point = int(len(data) * 2 / 3)
        data_train = data[:split_point]
        eval_len = int(len(data[split_point:]) / 2)
        data_test = data[split_point:split_point + eval_len]
        data_val = data[split_point + eval_len:]
        self.pat_info_test = data_test

        if MODEL in ['RETAIN']:
            self.train_loader = DataLoader(CustomDataset(data_train), batch_size=1, shuffle=True, \
                                           collate_fn=lambda x: collate_fn_RETAIN(x[0], self.voc_size))
            self.val_loader = DataLoader(CustomDataset(data_val), batch_size=1, shuffle=False, \
                                         collate_fn=lambda x: collate_fn_RETAIN(x[0], self.voc_size))
            self.test_loader = DataLoader(CustomDataset(data_test), batch_size=1, shuffle=False, \
                                          collate_fn=lambda x: collate_fn_RETAIN(x[0], self.voc_size))
            print("generated train/val/test dataloaders for RETAIN model!")

        elif MODEL in ['GAMENet']:
            self.train_loader = DataLoader(CustomDataset(data_train), batch_size=1, shuffle=True, \
                                           collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.val_loader = DataLoader(CustomDataset(data_val), batch_size=1, shuffle=False, \
                                         collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.test_loader = DataLoader(CustomDataset(data_test), batch_size=1, shuffle=False, \
                                          collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.ehr_adj = self._generate_ehr_adj_for_GAMENet(data_train)
            print("generated train/val/test dataloaders for GAMENet model!")

        elif MODEL in ['MICRON']:
            self.train_loader = DataLoader(CustomDataset(data_train), batch_size=1, shuffle=True, \
                                           collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.val_loader = DataLoader(CustomDataset(data_val), batch_size=1, shuffle=False, \
                                         collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.test_loader = DataLoader(CustomDataset(data_test), batch_size=1, shuffle=False, \
                                          collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            print("generated train/val/test dataloaders for MICRON model!")

        elif MODEL in ['SafeDrug']:
            # SafeDrug model needs a mapping to SMILES strings
            self.train_loader = DataLoader(CustomDataset(data_train), batch_size=1, shuffle=True, \
                                           collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.val_loader = DataLoader(CustomDataset(data_val), batch_size=1, shuffle=False, \
                                         collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.test_loader = DataLoader(CustomDataset(data_test), batch_size=1, shuffle=False, \
                                          collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.tool.add_new_code("SMILES")
            self.ddi_mask_H, SMILES = self._generate_ddi_mask_H_for_SafeDrug(self.tool.ATC4_to_SMILES)
            self.med_molecule_info = self._generate_med_molecule_info_for_SafeDrug(SMILES)
            print("generated train/val/test dataloaders for SafeDrug model!")