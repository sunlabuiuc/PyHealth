from importlib import reload

import ipdb
import numpy as np
import pandas as pd
import scipy


def _make_sim_noisy(n, nclass, seed, signal=0.1, p=None):
    np.random.seed(seed)
    labels = np.zeros((n, nclass))
    y = np.random.choice(np.arange(nclass), n, p=p)
    labels[np.arange(n), y] = 1

    def gen_one_pred(y):
        x = np.random.random(nclass)
        x[y] += signal * (3 if y <1 and nclass > 2 else 1)
        return scipy.special.softmax(x)

    preds = [gen_one_pred(y_i) for y_i in y]

    return np.asarray(preds), np.asarray(labels)

def _make_sim_real(n, nclass, seed, signal=3, **kwargs):
    probs, _ = _make_sim_noisy(n, nclass, seed=seed, signal=signal, **kwargs)
    np.random.seed(seed)
    classes = [i for i in range(nclass)]
    y = np.asarray([np.random.choice(classes, 1, p=probs[i])[0] for i in range(len(probs))])
    labels = np.zeros((n, nclass))
    labels[np.arange(n), y] = 1
    return probs, labels

def make_sim_data(n, nclass, seed, sim_type="", signal=3, **kwargs):
    if sim_type == 'noisy': return _make_sim_noisy(n,nclass,seed, signal, **kwargs)
    if sim_type == 'real': return _make_sim_real(n,nclass,seed, signal, **kwargs)


if __name__ == '__main__':
    import pyhealth.calib.predictionset.scrib as scrib
    reload(scrib)
    N, K = 10000, 5
    sim_type='real'
    valid_output, valid_y = make_sim_data(N, K, 123, sim_type=sim_type, signal=1.5)
    test_output, test_y = make_sim_data(N, K, 778, sim_type=sim_type, signal=1.5)
    r=0.3
    rks=[0.3, 0.3, 0.3, 0.3, 0.3]

    best_ts, best_loss = scrib._CoordDescent.search(valid_output, valid_y, rks, B=5, loss_func='classspec')
    best_ts, best_loss = scrib._CoordDescent.search(valid_output, valid_y, 0.3, B=5, loss_func='overall')#can be "classSpec" as well, in which case alphas should be list-like

    print(best_ts, best_loss)