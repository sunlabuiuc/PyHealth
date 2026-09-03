from pyhealth.datasets import SampleDataset


class SampleKGDataset(SampleDataset):
    """Sample KG dataset class.

    This class inherits from `SampleDataset` and is specifically designed
        for KG datasets.

    Args:
        samples: a list of samples 
        A sample is a dict containing following data:
        {
            'triple': a positive triple  e.g., (0, 0, 2835)
            'ground_truth_head': a list of ground truth of the head entity in the dataset given
                query (e.g., (?, 0, 2835)) with current relation r and tail entity t.
                e.g., [1027, 1293, 5264, 1564, 7416, 6434, 2610, 4094, 2717, 5007, 5277, 5949, 0, 6870, 6029]
            'ground_truth_tail': a list of ground truth of the tail entity in the dataset given
                query (e.g., (0, 0, ?)) with current head entity h and relation r.
                e.g., [398, 244, 3872, 3053, 1711, 2835, 1348, 2309]
            'subsampling_weight': the subsampling weight (a scalar) of this triple, which may be applied for loss calculation
        }
        dataset_name: the name of the dataset. Default is None.
        task_name: the name of the task. Default is None.

    Examples:
        >>> from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import SampleKGDataset
        >>> entity2id = {"aspirin": 0, "headache": 1, "ibuprofen": 2}
        >>> relation2id = {"treats": 0}
        >>> samples = [
        ...     {
        ...         "triple": (0, 0, 1),
        ...         "ground_truth_head": [0, 2],
        ...         "ground_truth_tail": [1],
        ...         "subsampling_weight": 1.0,
        ...     },
        ... ]
        >>> dataset = SampleKGDataset(
        ...     samples=samples,
        ...     dataset_name="toy_kg",
        ...     entity_num=len(entity2id),
        ...     relation_num=len(relation2id),
        ...     entity2id=entity2id,
        ...     relation2id=relation2id,
        ... )
        >>> dataset.stat()
    """
    def __init__(
        self, 
        samples, 
        dataset_name="", 
        task_name="", 
        dev=False,  
        entity_num=0,
        relation_num=0,
        entity2id=None,
        relation2id=None,
        **kwargs
        ):

        # SampleDataset's __init__ now expects a path to a directory built
        # by SampleBuilder.save() (it's litdata-backed), not an in-memory
        # list of samples. SampleKGDataset predates that change and still
        # works with samples held directly in memory, so we set the base
        # attributes ourselves instead of delegating to super().__init__().
        self.samples = samples
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.dev = dev
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.sample_size = len(samples)
        self.task_spec_param = None
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in relation2id.items()}
        if kwargs != None:
            self.task_spec_param = kwargs

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.sample_size

    def __getitem__(self, index):
        """
        A sample is a dict containing following data:
        {
            'triple': a positive triple  e.g., (0, 0, 2835)
            'ground_truth_head': a list of ground truth of the head entity in the dataset given
                query (e.g., (?, 0, 2835)) with current relation r and tail entity t.
                e.g., [1027, 1293, 5264, 1564, 7416, 6434, 2610, 4094, 2717, 5007, 5277, 5949, 0, 6870, 6029]
            'ground_truth_tail': a list of ground truth of the tail entity in the dataset given
                query (e.g., (0, 0, ?)) with current head entity h and relation r.
                e.g., [398, 244, 3872, 3053, 1711, 2835, 1348, 2309]
            'subsampling_weight': the subsampling weight (a scalar) of this triple, which may be applied for loss calculation
        }
        """
        return self.samples[index]
    
    def stat(self):
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of base dataset (dev={self.dev}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of triples: {len(self.samples)}")
        lines.append(f"\t- Number of entities: {self.entity_num}")
        lines.append(f"\t- Number of relations: {self.relation_num}")
        lines.append(f"\t- Task name: {self.task_name}")
        lines.append(f"\t- Task-specific hyperparameters: {self.task_spec_param}")
        lines.append("")
        print("\n".join(lines))
        return 
