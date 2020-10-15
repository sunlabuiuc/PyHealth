import numpy as np
from numpy import random

def generate_label(n_sample, task = 'binaryclass', n_class = 2):

    if task == 'binaryclass':
        assert n_class == 2, 'default value 2 for n_class'
        label_data = random.randint(0,2,n_sample)
    elif task == 'multiclass':
        assert n_class > 2, 'n_class>2 for multiclass task'
        label_data = np.zeros((n_sample, n_class))
        lv = random.randint(0,n_class,n_sample)
        for idx, value in enumerate(lv):
            label_data[idx, value] = 1
    elif task == 'multilabel':
        assert n_class > 2, 'n_class>2 for multilabel task'
        label_data = random.randint(0,2,n_sample*n_class).reshape(-1, n_class)
        lv = random.randint(0,n_class,n_sample)
        for idx, value in enumerate(lv):
            label_data[idx, value] = 1
    elif task == 'regression':
        label_data =  random.random(n_sample)*10
    else:
        raise 'fill in correct parameter [task], current support in [binary, multiclass, multilabel, regression]'
    return label_data

class generate_simulation_sequence_data:
        
    def __init__(self, 
                 n_sample = 100,
                 n_maxseq = 15,
                 n_feat = 20, 
                 task = 'binaryclass',
                 n_class = 2):
        self.n_sample = n_sample
        self.n_feat = n_feat
        self.n_maxseq = n_maxseq
        self.task = task
        self.n_class = n_class
        self._genearte_data()

    def _genearte_data(self):
        feat_data = []
        seqlen_data = []
        time_data = []
        for idx in range(self.n_sample):
            cur_seqlen = random.randint(2, self.n_maxseq)
            seqlen_data.append(cur_seqlen)
            feat_data.append(random.random((cur_seqlen, self.n_feat)))
            time_data.append(np.cumsum(random.random(cur_seqlen)*10))
        label_data = generate_label(self.n_sample, self.task, self.n_class)
        self.data = {
          'x': feat_data,
          't': time_data,
          'l': seqlen_data,
          'y': label_data,
        }

    def __call__(self):
        return self.data

    def get_data(self):
        return self.data

class generate_simulation_image_data:

    def __init__(self, 
                 n_sample = 100, 
                 n_width = 224, 
                 n_height = 224,
                 task = 'binaryclass', 
                 n_class = 2):
        self.n_sample = n_sample
        self.n_width = n_width
        self.n_height = n_height
        self.task = task
        self.n_class = n_class
        self._genearte_data()

    def _genearte_data(self):
        feat_data = np.array(random.randint(0,256,(self.n_sample, 1, self.n_width, self.n_height)),dtype='uint8')
        label_data = generate_label(self.n_sample, self.task, self.n_class)
        self.data = {
          'x': feat_data,
          'y': label_data,
        }
    def __call__(self):
        return self.data

    def get_data(self):
        return self.data

class generate_simulation_ecg_data:

    def __init__(self, 
                 n_sample = 100, 
                 n_length = 2000, 
                 task = 'binaryclass', 
                 n_class = 2):
        self.n_sample = n_sample
        self.n_length = n_length
        self.task = task
        self.n_class = n_class
        self._genearte_data()
        
    def _genearte_data(self):
        feat_data = random.random((self.n_sample, self.n_length))
        label_data = generate_label(self.n_sample, self.task, self.n_class)
        self.data = {
          'x': feat_data,
          'y': label_data,
        }
    def __call__(self):
        return self.data

    def get_data(self):
        return self.data

if __name__ == '__main__':
    data = generate_simulation_sequence_data(n_sample = 10)()
    print (data.keys())
    data = generate_simulation_image_data(n_sample = 10)()
    print (data.keys())
    data = generate_simulation_ecg_data(n_sample = 10)()
    print (data.keys())
