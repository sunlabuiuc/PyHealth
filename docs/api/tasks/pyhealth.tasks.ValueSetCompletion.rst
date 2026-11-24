pyhealth.tasks.ValueSetCompletion
=================================

Retrieval-augmented set completion (RASC) for clinical value set authoring: retrieve the K value sets with the most similar titles, pool their codes as candidates, and classify each candidate as in or out of the target value set. Build the retrieval index once with :func:`~pyhealth.tasks.build_retrieval_index`.

.. autofunction:: pyhealth.tasks.build_retrieval_index

.. autofunction:: pyhealth.tasks.value_set_completion.sapbert_encoder

.. autoclass:: pyhealth.tasks.ValueSetCompletion
    :members:
    :undoc-members:
    :show-inheritance:
