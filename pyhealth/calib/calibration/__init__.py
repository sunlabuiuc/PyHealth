"""Model calibration methods"""
from pyhealth.calib.calibration.dircal import DirichletCalibration
from pyhealth.calib.calibration.hb import HistogramBinning
from pyhealth.calib.calibration.kcal import KCal
from pyhealth.calib.calibration.logistic_recalibration import LogisticRecalibration
from pyhealth.calib.calibration.temperature_scale import TemperatureScaling

__all__ = ['DirichletCalibration', 'HistogramBinning', 'KCal', 'LogisticRecalibration',
           'TemperatureScaling']
