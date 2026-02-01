# PyHealth Calibration Module

This module contains post-hoc uncertainty quantification methods for healthcare AI models, including model calibration methods and prediction set constructors.

## Model Calibration Methods

Model calibration methods adjust predicted probabilities to better reflect true confidence levels.

### Temperature Scaling

**Modes**: `multiclass`, `multilabel`, `binary`

**Class**: `pyhealth.calib.calibration.TemperatureScaling`

Temperature scaling (also known as Platt scaling for binary classification) is a simple yet effective calibration method that scales logits by a learned temperature parameter.

**Guarantee**: Empirically reduces Expected Calibration Error (ECE). No formal finite-sample statistical guarantee, but widely effective in practice for improving probability calibration.

**Reference**:
- Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. "On calibration of modern neural networks." ICML 2017.

### Histogram Binning

**Modes**: `multiclass`, `multilabel`, `binary`

**Class**: `pyhealth.calib.calibration.HistogramBinning`

Histogram binning is a non-parametric calibration method that bins predictions and adjusts probabilities within each bin.

**Guarantee**: Asymptotically consistent calibration as calibration set size → ∞. Provides better empirical calibration (lower ECE) than uncalibrated models. For top-label calibration, provides distribution-free top-label calibration guarantees.

**References**:
- Zadrozny, Bianca, and Charles Elkan. "Learning and making decisions when costs and probabilities are both unknown." In Proceedings of the seventh ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 204-213. 2001.
- Gupta, Chirag, and Aaditya Ramdas. "Top-label calibration and multiclass-to-binary reductions." ICLR 2022.

### Dirichlet Calibration

**Modes**: `multiclass`

**Class**: `pyhealth.calib.calibration.DirichletCalibration`

Dirichlet calibration learns a matrix transformation of logits with regularization for improved calibration.

**Guarantee**: More expressive than temperature scaling. Empirically reduces multiclass calibration error (ECE, classwise-ECE) by learning class-specific transformations. Optimizes log-likelihood under Dirichlet prior.

**Reference**:
- Kull, Meelis, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach. "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration." NeurIPS 2019.

### KCal (Kernel-Based Calibration)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.calibration.KCal`

KCal uses kernel density estimation on embeddings for full multiclass calibration. The model must support `embed=True` in forward pass.

**Guarantee**: Leverages learned representations for calibration. Empirically reduces ECE, particularly effective when embedding space captures semantic structure. Provides non-parametric calibration through kernel density estimation in embedding space.

**Reference**:
- Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun. "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration for Deep Neural Networks." ICLR 2023.

## Prediction Set Methods (Conformal Prediction)

Conformal prediction is a framework for constructing prediction sets with formal statistical coverage guarantees. Instead of producing a single prediction, these methods output a set of plausible labels that is guaranteed to contain the true label with a user-specified probability (e.g., 90% or 95%). The key advantage is that these guarantees are **distribution-free** and hold for finite samples without assumptions on the data distribution or model—only requiring that calibration and test data are exchangeable (i.e., drawn from the same distribution).

For example, with α=0.1 (90% coverage), conformal prediction guarantees that P(Y ∈ C(X)) ≥ 0.9, where C(X) is the prediction set for input X. This is particularly valuable in high-stakes applications like healthcare, where quantifying uncertainty is critical for safe decision-making.

### BaseConformal (Base Split Conformal Prediction)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.predictionset.BaseConformal`

BaseConformal implements standard split conformal prediction without covariate shift correction. It provides a clean baseline implementation for constructing prediction sets with distribution-free coverage guarantees by calibrating score thresholds on a held-out calibration set.

**Guarantee**: Distribution-free finite-sample coverage under exchangeability:
- **Marginal**: P(Y ∉ C(X)) ≤ α (with high probability)
- **Class-conditional**: P(Y ∉ C(X) | Y=k) ≤ α_k for each class k

No assumptions on the model or data distribution required (only exchangeability).

**References**:
- Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer. "Algorithmic learning in a random world." Springer, 2005.
- Lei, Jing, et al. "Distribution-free predictive inference for regression." Journal of the American Statistical Association (2018).

### LABEL (Least Ambiguous Set-valued Classifier)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.predictionset.LABEL`

LABEL is a conformal prediction method that constructs prediction sets with bounded error levels. Supports both marginal and class-conditional coverage.

**Guarantee**: Distribution-free finite-sample coverage guarantees:
- **Marginal**: P(Y ∉ C(X)) ≤ α
- **Class-conditional**: P(Y ∉ C(X) | Y=k) ≤ α_k for each class k

Constructs least ambiguous (minimal size) sets subject to coverage constraints. Similar to BaseConformal but optimized for minimal ambiguity.

**Reference**:
- Sadinle, Mauricio, Jing Lei, and Larry Wasserman. "Least ambiguous set-valued classifiers with bounded error levels." Journal of the American Statistical Association 114, no. 525 (2019): 223-234.

### SCRIB (Set-classifier with Class-specific Risk Bounds)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.predictionset.SCRIB`

SCRIB controls class-specific risk while minimizing prediction set ambiguity through optimized class-specific thresholds.

**Guarantee**: Class-specific risk control with minimal ambiguity:
- **Overall**: P(Y ∉ C(X) | |C(X)|=1) ≤ risk (error rate on singleton predictions)
- **Class-specific**: P(Y ∉ C(X) | Y=k, |C(X)|=1) ≤ risk_k for each class k

Optimizes class-specific thresholds via coordinate descent to minimize prediction set ambiguity while respecting risk bounds.

**Reference**:
- Lin, Zhen, Lucas Glass, M. Brandon Westover, Cao Xiao, and Jimeng Sun. "SCRIB: Set-classifier with Class-specific Risk Bounds for Blackbox Models." AAAI 2022.

### FavMac (Fast Value-Maximizing Prediction Sets)

**Modes**: `multilabel`

**Class**: `pyhealth.calib.predictionset.FavMac`

FavMac constructs prediction sets that maximize value while controlling cost/risk, particularly useful for multilabel classification with asymmetric costs.

**Guarantee**: Conformal cost control with value maximization:
- **Expected cost**: E[Cost(C(X), Y)] ≤ target_cost (in expectation over calibration)
- **Adaptive thresholds**: Dynamically adjusts thresholds online to control false positive rates

Particularly useful for multilabel tasks with asymmetric costs (e.g., medical diagnosis where false positives/negatives have different costs).

**References**:
- Lin, Zhen, Shubhendu Trivedi, Cao Xiao, and Jimeng Sun. "Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control (FavMac)." ICML 2023.
- Fisch, Adam, Tal Schuster, Tommi Jaakkola, and Regina Barzilay. "Conformal prediction sets with limited false positives." ICML 2022.

### CovariateLabel (Covariate Shift Adaptive Conformal)

**Modes**: `multiclass`

**Class**: `pyhealth.calib.predictionset.CovariateLabel`

CovariateLabel extends LABEL to handle covariate shift between calibration and test distributions using likelihood ratio weighting. The default KDE-based approach follows the CoDrug method, which uses kernel density estimation on embeddings to compute likelihood ratios. Users can also provide custom weights for flexibility.

**Guarantee**: Distribution-free coverage under covariate shift:
- **Marginal**: P_test(Y ∉ C(X)) ≤ α on test distribution
- **Class-conditional**: P_test(Y ∉ C(X) | Y=k) ≤ α_k on test distribution

Uses importance weighting (likelihood ratios w(x) = p_test(x)/p_cal(x)) to correct for distribution shift between calibration and test sets. Valid when weights are well-estimated. Supports KDE-based automatic weighting (CoDrug) or custom user-provided weights.

**References**:
- Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and Aaditya Ramdas. "Conformal prediction under covariate shift." NeurIPS 2019. https://arxiv.org/abs/1904.06019
- Laghuvarapu, Siddhartha, Zhen Lin, and Jimeng Sun. "Conformal Drug Property Prediction with Density Estimation under Covariate Shift." NeurIPS 2023. https://arxiv.org/abs/2310.12033

## Usage

See the [full documentation](https://pyhealth.readthedocs.io/en/latest/api/calib.html) for detailed API references and examples.