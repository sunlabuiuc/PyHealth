pyhealth.metrics.ecg_delineation
===============================

ECG delineation metrics with explicit annotation ignore masks.
All point metrics pool the confusion matrix across valid samples. Precision,
recall, F1 and point IoU are macro-averaged over the fixed four LUDB classes,
including background; absent-class zero denominators score zero. Accuracy is
the fraction of correctly labeled supervised points. Ignored points never
contribute to confusion counts, loss or segment interiors.

The segment metric is the research runner's **custom greedy definition**:
contiguous same-class intervals; descending-IoU one-to-one matching; stable
index tie-breaking; even zero-overlap candidate pairs may be matched;
unmatched intervals add zeros; class scores pool across sequences, then
macro-average all four classes. This is not optimal matching, onset/offset
tolerance F1, or a metric verified equivalent to the paper's reported mIoU.
Do not compare it directly against the paper without a metric-definition audit.

Ignored points and gaps break segments. Reconstruct only adjacent,
non-overlapping test windows from the same patient and lead; do not flatten
across patients/leads. The helper rejects overlaps because no aggregation
policy has been chosen for them. Point metrics remain window-based.

Evaluation loss comes from non-ignored cross-entropy. The supplied example
reports an **unweighted mean of non-empty batch losses**, as in the historical
runner, not a globally point-weighted mean. All-ignored windows are counted but
add no evaluation loss. No target information is inferred from predictions.

.. autofunction:: pyhealth.metrics.ecg_delineation.ecg_delineation_metrics

.. autofunction:: pyhealth.metrics.ecg_delineation.segment_mean_iou

.. autofunction:: pyhealth.metrics.ecg_delineation.assemble_ecg_sequences
