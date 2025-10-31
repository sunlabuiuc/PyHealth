# PyHealth Calibration Module

This module contains post-hoc uncertainty quantification methods for healthcare AI models, including model calibration methods and prediction set constructors.

## Model Calibration Methods

Model calibration methods adjust predicted probabilities to better reflect true confidence levels.

### Temperature Scaling

**Modes**: `multiclass`, `multilabel`, `binary`

**Class**: `pyhealth.calib.calibration.TemperatureScaling`

Temperature scaling (also known as Platt scaling for binary classification) is a simple yet effective calibration method that scales logits by a learned temperature parameter.

**Reference**:
- Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. "On calibration of modern neural networks." ICML 2017.

### Histogram Binning

**Modes**: `multiclass`, `multilabel`, `binary`

**Class**: `pyhealth.calib.calibration.HistogramBinning`

Histogram binning is a non-parametric calibration method that bins predictions and adjusts probabilities within each bin.

**References**:
- Zadrozny, Bianca, and Charles Elkan. "Learning and making decisions when costs and probabilities are both unknown." In Proceedings of the seventh ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 204-213. 2001.
- Gupta, Chirag, and Aaditya Ramdas. "Top-label calibration and multiclass-to-binary reductions." ICLR 2022.

### Dirichlet Calibration

**Modes**: `multiclass`

**Class**: `pyhealth.calib.calibration.DirichletCalibration`

Dirichlet calibration learns a matrix transformation of logits with regularization for improved calibration.

**Reference**:
- Kull, Meelis, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach. "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration." NeurIPS 2019.

### KCal (Kernel-Based Calibration)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.calibration.KCal`

KCal uses kernel density estimation on embeddings for full multiclass calibration. The model must support `embed=True` in forward pass.

**Reference**:
- Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun. "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration for Deep Neural Networks." ICLR 2023.

## Prediction Set Methods

Prediction set methods provide set-valued predictions with statistical coverage guarantees.

### LABEL (Least Ambiguous Set-valued Classifier)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.predictionset.LABEL`

LABEL is a conformal prediction method that constructs prediction sets with bounded error levels. Supports both marginal and class-conditional coverage.

**Reference**:
- Sadinle, Mauricio, Jing Lei, and Larry Wasserman. "Least ambiguous set-valued classifiers with bounded error levels." Journal of the American Statistical Association 114, no. 525 (2019): 223-234.

### SCRIB (Set-classifier with Class-specific Risk Bounds)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.predictionset.SCRIB`

SCRIB controls class-specific risk while minimizing prediction set ambiguity through optimized class-specific thresholds.

**Reference**:
- Lin, Zhen, Lucas Glass, M. Brandon Westover, Cao Xiao, and Jimeng Sun. "SCRIB: Set-classifier with Class-specific Risk Bounds for Blackbox Models." AAAI 2022.

### FavMac (Fast Value-Maximizing Prediction Sets)

**Modes**: `multilabel`

**Class**: `pyhealth.calib.predictionset.FavMac`

FavMac constructs prediction sets that maximize value while controlling cost/risk, particularly useful for multilabel classification with asymmetric costs.

**References**:
- Lin, Zhen, Shubhendu Trivedi, Cao Xiao, and Jimeng Sun. "Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control (FavMac)." ICML 2023.
- Fisch, Adam, Tal Schuster, Tommi Jaakkola, and Regina Barzilay. "Conformal prediction sets with limited false positives." ICML 2022.

### CovariateLabel (Covariate Shift Adaptive Conformal)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.predictionset.CovariateLabel`

CovariateLabel extends LABEL to handle covariate shift between calibration and test distributions using likelihood ratio weighting.

**Reference**:
- Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and Aaditya Ramdas. "Conformal prediction under covariate shift." NeurIPS 2019.

## Usage

See the [full documentation](https://pyhealth.readthedocs.io/en/latest/api/calib.html) for detailed API references and examples.