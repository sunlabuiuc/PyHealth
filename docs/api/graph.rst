Graph
=====

The ``pyhealth.graph`` module lets you bring a healthcare knowledge graph into
your PyHealth pipeline. Graph-based models like GraphCare and GNN can use
relational medical knowledge — drug interactions, disease hierarchies,
symptom–diagnosis links — to enrich patient representations beyond what
raw EHR codes alone can capture.

What Is a Knowledge Graph?
---------------------------

A knowledge graph encodes medical relationships as **(head, relation, tail)**
triples. For example:

- ``("aspirin", "treats", "headache")``
- ``("metformin", "used_for", "type_2_diabetes")``
- ``("ICD9:250", "is_a", "ICD9:249")``

PyHealth does not ship a built-in graph — you bring triples from a source of
your choice (UMLS, DrugBank, an ICD hierarchy, a custom ontology, etc.) and
the :class:`~pyhealth.graph.KnowledgeGraph` class handles indexing, entity
mappings, and k-hop subgraph extraction. The typical use case is querying the
graph at training time: given a patient's active codes, extract the local
subgraph around those codes and feed it to a graph-aware model.

Getting Started
---------------

The simplest way to create a graph is to pass a list of triples directly:

.. code-block:: python

    from pyhealth.graph import KnowledgeGraph

    triples = [
        ("aspirin",    "treats",     "headache"),
        ("headache",   "symptom_of", "migraine"),
        ("ibuprofen",  "treats",     "headache"),
    ]
    kg = KnowledgeGraph(triples=triples)
    kg.stat()
    # KnowledgeGraph: 4 entities, 2 relations, 3 triples

For larger graphs it is more practical to load from a CSV or TSV file. The
file should have columns named ``head``, ``relation``, and ``tail``:

.. code-block:: python

    kg = KnowledgeGraph(triples="path/to/medical_kg.tsv")

Exploring the Graph
-------------------

Once built, you can inspect the graph and look up neighbours for any entity:

.. code-block:: python

    kg.num_entities     # total unique entities
    kg.num_relations    # total unique relation types
    kg.num_triples      # total edges

    kg.has_entity("aspirin")     # True / False
    kg.neighbors("aspirin")      # list of (relation, tail) pairs

    # Integer ID mappings used internally by PyG
    kg.entity2id["aspirin"]      # → int
    kg.id2entity[0]              # → entity name string

Extracting Patient Subgraphs
------------------------------

The main reason to build a knowledge graph is to extract a patient-specific
subgraph at training time. ``subgraph()`` returns all entities reachable
within *n* hops of a set of seed codes, as a PyTorch Geometric ``Data``
object:

.. code-block:: python

    patient_codes = ["ICD9:250.00", "NDC:0069-0105"]
    subgraph = kg.subgraph(seed_entities=patient_codes, num_hops=2)

.. note::

   ``subgraph()`` requires `PyTorch Geometric <https://pyg.org/>`_
   (``torch_geometric``). The graph can still be constructed and explored
   without it — only subgraph extraction needs PyG.

   Install with: ``pip install torch-geometric``

Using with GraphProcessor in a Task
-------------------------------------

To feed subgraphs into a model automatically during data loading, pass a
configured :class:`~pyhealth.processors.GraphProcessor` instance in your
task's ``input_schema``. The processor will call ``kg.subgraph()`` for each
patient sample:

.. code-block:: python

    from pyhealth.graph import KnowledgeGraph
    from pyhealth.processors import GraphProcessor
    from pyhealth.tasks import BaseTask

    kg = KnowledgeGraph(triples="medical_kg.tsv")

    class MyGraphTask(BaseTask):
        task_name = "MyGraphTask"
        input_schema = {
            "conditions":  "sequence",
            "kg_subgraph": GraphProcessor(kg, num_hops=2),
        }
        output_schema = {"label": "binary"}

        def __call__(self, patient):
            ...

Pre-computed Node Embeddings
-----------------------------

If you already have entity embeddings (e.g. from TransE or an LLM), you can
attach them to the graph at construction time. The model can then use these
as initial node features instead of learning them from scratch:

.. code-block:: python

    import torch

    node_features = torch.randn(kg.num_entities, 64)  # (num_entities, feat_dim)
    kg = KnowledgeGraph(triples=triples, node_features=node_features)

API Reference
-------------

.. toctree::
    :maxdepth: 3

    graph/pyhealth.graph.KnowledgeGraph
