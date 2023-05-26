"""Model calibration methods"""
from pyhealth.calib.calibration.dircal import DirichletCalibration
from pyhealth.calib.calibration.hb import HistogramBinning
from pyhealth.calib.calibration.kcal import KCal
from pyhealth.calib.calibration.temperature_scale import TemperatureScaling

__all__ = ['DirichletCalibration', 'HistogramBinning', 'KCal', 'TemperatureScaling']
