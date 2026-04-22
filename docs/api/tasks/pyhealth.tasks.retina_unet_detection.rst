Retina U-Net Detection Task
===========================

The RetinaUNetDetectionTask converts segmentation masks into object detection
targets (bounding boxes and labels) while preserving segmentation supervision.

This task is designed to support Retina U-Net style training pipelines.

Example
-------

.. code-block:: python

    import numpy as np
    from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask

    task = RetinaUNetDetectionTask()

    image = np.random.rand(128, 128, 3)
    mask = np.zeros((128, 128))
    mask[30:60, 40:80] = 1

    sample = {"image": image, "mask": mask}
    output = task.process_sample(sample)

    print(output["boxes"])

API Reference
-------------

.. automodule:: pyhealth.tasks.retina_unet_detection
   :members:
   :undoc-members:
   :show-inheritance:
