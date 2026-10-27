pyhealth.interpret.methods.base_interpreter
===========================================

Overview
--------

:class:`pyhealth.interpret.methods.BaseInterpreter` is the abstract base class
shared by every attribution method in ``pyhealth.interpret.methods``. Subclasses
implement ``attribute(**batch)``, which returns a dictionary of attribution
tensors keyed by the task's feature keys. Subclass it to add a new
interpretability method.

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.BaseInterpreter
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
