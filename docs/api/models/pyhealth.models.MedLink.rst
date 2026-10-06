pyhealth.models.MedLink
===================================

The complete MedLink model.

.. note::

   ``pyhealth.models.medlink.utils.collate_fn`` (used by
   ``get_train_dataloader``) drops the ``s_n`` (hard negative) field for an
   entire batch if any sample in it lacks one, rather than producing a
   partially-present, misaligned list -- ``MedLink.forward`` consumes
   ``s_n`` as a whole-batch field (``corpus = s_p + s_n``), so a
   per-sample-optional value would corrupt that concatenation.

.. autoclass:: pyhealth.models.MedLink
    :members:
    :undoc-members:
    :show-inheritance: