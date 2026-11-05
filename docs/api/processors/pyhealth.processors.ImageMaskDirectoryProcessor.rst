pyhealth.processors.ImageMaskDirectoryProcessor
===============================================

Processor for a directory of 2D image mask slices in any PIL-readable
format. The default glob is ``*.png``; pass ``pattern`` to use another
lossless format such as ``*.tif``. JPEG masks are lossy and corrupt integer
labels. The processor stacks the slices into a 3D integer mask volume and can
align them with a reference DICOM series.

.. autoclass:: pyhealth.processors.ImageMaskDirectoryProcessor
    :members:
    :undoc-members:
    :show-inheritance:
