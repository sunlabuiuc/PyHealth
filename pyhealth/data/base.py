# -*- coding: utf-8 -*-
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Standard_Template(object):
    """Abstract Class which can be inherited by various datasets,
    Key information and memory friendly information will be saved in the
    data dictionary. Otherwise, save the event and sequence location instead.
    """

    def __init__(self, patient_id):
        self.data = {}
        self.data['patient_id'] = str(patient_id)
        self.data['admission_list'] = []

    @abc.abstractmethod
    def parse_patient(self, pd_series, mapping_dict=None):
        pass

    @abc.abstractmethod
    def parse_admission(self, pd_df, mapping_dict=None):
        pass

    # @abc.abstractmethod
    def parse_icu(self, pd_df, mapping_dict=None):
        pass
