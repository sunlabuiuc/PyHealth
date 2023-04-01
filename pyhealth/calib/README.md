This module contains several post-hoc uncertainty quantification methods.

# Model Calibration:

### KCal

Used in: `multiclass`.

The model needs to be able to take `embed=True` in `forward` and output the penultimate embedding in the output.

Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun.
"Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration for Deep Neural Networks."
ICLR 2023.

### Temperature Scaling

Used in: `multiclass`, `multilabel` and `binary`.

Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger.
"On calibration of modern neural networks."
ICML 2017.

### Histogram Binning

Used in: `multiclass`, `multilabel` and `binary`.


Zadrozny, Bianca, and Charles Elkan.
"Learning and making decisions when costs and probabilities are both unknown."
In Proceedings of the seventh ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 204-213. 2001.

Gupta, Chirag, and Aaditya Ramdas.
"Top-label calibration and multiclass-to-binary reductions."
ICLR 2022.

# Prediction Set:

### SCRIB

Used in: `multiclass`.

Lin, Zhen, Lucas Glass, M. Brandon Westover, Cao Xiao, and Jimeng Sun.
"SCRIB: Set-classifier with Class-specific Risk Bounds for Blackbox Models."
AAAI 2022.

### LABEL

Used in: `multiclass`.

Sadinle, Mauricio, Jing Lei, and Larry Wasserman.
"Least ambiguous set-valued classifiers with bounded error levels."
Journal of the American Statistical Association 114, no. 525 (2019): 223-234.