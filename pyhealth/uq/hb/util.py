import numpy as np


def get_uniform_mass_bins(probs, n_bins):
    assert (probs.size >= n_bins), "Fewer points than bins"

    probs_sorted = np.sort(probs)

    # split probabilities into groups of approx equal size
    groups = np.array_split(probs_sorted, n_bins)
    bin_edges = list()
    bin_upper_edges = list()

    for cur_group in range(n_bins - 1):
        bin_upper_edges += [max(groups[cur_group])]
    bin_upper_edges += [np.inf]

    return np.array(bin_upper_edges)


def bin_points(scores, bin_edges):
    assert (bin_edges is not None), "Bins have not been defined"
    scores = scores.squeeze()
    assert (np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)


def nudge(matrix, delta):
    return ((matrix + np.random.uniform(low=0,
                                        high=delta,
                                        size=(matrix.shape))) / (1 + delta))


class identity():
    def predict_proba(self, x):
        return x

    def predict(self, x):
        return np.argmax(x, axis=1)
