import abc
import numpy as np


def _enumerate_index(indices: np.ndarray):
    """
    Replaces the i^{th} smallest value in the indices list by i for each i
    :param indices: A one-dimensional array containing the original indices
    :return new_indices: An array of same shape as indices, modified as described above
    """
    assert len(indices.shape) == 1, f'indices must be a one dimensional array but found shape {indices.shape}'
    unique_vals = np.unique(indices)
    old2new_map = dict((unique_vals[i], i) for i in range(unique_vals.shape[0]))
    return np.asarray([old2new_map[indices[i]] for i in range(indices.shape[0])])


class Oracle(abc.ABC):
    """
    An abstract class defining an oracle that answers same-cluster queries
    """

    def __init__(self, num_items: int, clusters: np.ndarray):
        """
        :param num_items: Number of items
        :param clusters: A one dimensional array containing the cluster index for each item
        """
        self._num_items = num_items
        self._clusters = _enumerate_index(clusters)
        self._num_clusters = np.max(self._clusters) + 1

    @abc.abstractmethod
    def sample(self, item_pairs: list[tuple[int, int]]) -> list[float]:
        """
        Query the given item pairs
        :param item_pairs: A list of item pairs (item_i, item_j)
        :return output: A list of same size as item_pairs containing the output for each item pair
        """
        raise NotImplementedError('Method sample not implemented')

    def config(self) -> dict:
        return {'num_items': self._num_items, 'num_clusters': self._num_clusters, 'clusters': self._clusters}


class PPOracle(Oracle):
    """
    An oracle based on the planted partition model
    """

    def __init__(self, num_items: int, clusters: np.ndarray, p: float, q: float):
        """
        :param num_items: Number of items
        :param clusters: A one dimensional array containing the cluster index for each item
        :param p: The probability with which items in the same cluster return one
        :param q: The probability with which items in the different clusters return one
        """
        super(PPOracle, self).__init__(num_items=num_items, clusters=clusters)
        self._p = p
        self._q = q

    def sample(self, item_pairs: list[tuple[int, int]]) -> list[float]:
        """
        Query the given item pairs
        :param item_pairs: A list of item pairs (item_i, item_j)
        :return output: A list of same size as item_pairs containing the output for each item pair
        """
        probs = np.asarray([self._p if self._clusters[item_i] == self._clusters[item_j] else self._q
                            for item_i, item_j in item_pairs])
        return np.asarray(np.random.random(size=probs.shape) <= probs).astype(float).tolist()

    def config(self) -> dict:
        config = super().config()
        config['p'] = self._p
        config['q'] = self._q
        config['name'] = 'PPOracle'
        return config


class PerturbedPPOracle(Oracle):
    """
    An oracle based on the planted partition model where p and q values are slightly perturbed for each item pair
    """

    def __init__(self, num_items: int, clusters: np.ndarray, p: float, q: float, perturbation_noise: float):
        """
        :param num_items: Number of items
        :param clusters: A one dimensional array containing the cluster index for each item
        :param p: The probability with which items in the same cluster return one
        :param q: The probability with which items in the different clusters return one
        :param perturbation_noise: Noise with which to perturb the p and q values
        """
        super(PerturbedPPOracle, self).__init__(num_items=num_items, clusters=clusters)
        self._p = p
        self._q = q
        self._perturbation_noise = perturbation_noise

        # Calculate the probabilities
        self._probs = np.zeros((self._num_items, self._num_items))
        perturbations = np.random.uniform(
            low=-self._perturbation_noise, high=self._perturbation_noise, size=(self._num_items, self._num_items)
        )
        perturbations = np.triu(perturbations, k=1) + np.triu(perturbations, k=1).T + np.diag(np.diag(perturbations))
        for item_i in range(self._num_items):
            for item_j in range(self._num_items):
                base_prob = self._p if self._clusters[item_i] == self._clusters[item_j] else self._q
                self._probs[item_i, item_j] = np.clip(base_prob + perturbations[item_i, item_j], 0.05, 0.95)

    def sample(self, item_pairs: list[tuple[int, int]]) -> list[float]:
        """
        Query the given item pairs
        :param item_pairs: A list of item pairs (item_i, item_j)
        :return output: A list of same size as item_pairs containing the output for each item pair
        """
        probs = np.asarray([self._probs[item_i, item_j] for item_i, item_j in item_pairs])
        return np.asarray(np.random.random(size=probs.shape) <= probs).astype(float).tolist()

    def config(self) -> dict:
        config = super().config()
        config['p'] = self._p
        config['q'] = self._q
        config['probs'] = self._probs
        config['perturbation_noise'] = self._perturbation_noise
        config['name'] = 'PerturbedPPOracle'
        return config
