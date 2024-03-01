import abc
import numpy as np

from typing import Optional


class Algorithm(abc.ABC):
    """
    An abstract class defining an algorithm for query-based clustering
    """

    def __init__(self, num_items: int, delta: float, num_clusters: Optional[int] = None):
        """
        :param num_items: Number of items
        :param delta: Confidence parameter
        :param num_clusters: Number of clusters to find
        """
        self._num_items = num_items
        self._clusters = np.zeros((self._num_items,))
        self._num_clusters = num_clusters
        self._delta = delta

    @abc.abstractmethod
    def select(self) -> list[tuple[int, int]]:
        """
        Returns a list containing pairs of items to be sampled at the current step
        :return item_pairs: A list of the form [(item_i1, item_j1), (item_i2, item_j2), ...]
        """
        raise NotImplementedError('Function select not implemented')

    @abc.abstractmethod
    def update(self, item_pairs: list[tuple[int, int]], observations: list[float]) -> None:
        """
        Updates the algorithm parameters using the current observations
        :param item_pairs: A list containing item pairs that were sampled
        :param observations: A list of same size as item_pairs containing the observation for each pair
        """
        raise NotImplementedError('Function update not implemented')

    @abc.abstractmethod
    def terminate(self) -> bool:
        """
        :return terminate: Is the algorithm ready to terminate?
        """
        raise NotImplementedError('Function terminate not implemented')

    @abc.abstractmethod
    def progress(self) -> float:
        """
        :return progress: A measure of progress between 0 and 1
        """
        raise NotImplementedError('Function progress not implemented')

    def result(self) -> dict[str]:
        """
        :return output: A dictionary containing required output at the end of the algorithm
        """
        assert self.terminate(), 'Algorithm has not terminated yet and hence results are not ready'
        return {'clusters': self._clusters}

    def config(self) -> dict:
        return {'num_items': self._num_items, 'num_clusters': self._num_clusters, 'delta': self._delta}


class SplitItems(Algorithm, abc.ABC):
    """
    An algorithm that uses confidence intervals to split items into two clusters
    """

    def __init__(
            self,
            num_items: int,
            delta: float,
            init_hat_p: Optional[float] = None,
            init_hat_q: Optional[float] = None
    ):
        """
        :param num_items: Number of items to consider
        :param delta: Confidence parameter
        :param init_hat_p: An initial value of hat_p if available
        :param init_hat_q: An initial value of hat_q if available
        """
        super(SplitItems, self).__init__(num_items=num_items, delta=delta)

        # Initialize time
        self._time = 0

        # Select the anchor and mark all other items as unassigned
        self._anchor = 0    # Always choose the first node as an anchor
        self._clusters[self._anchor] = 0  # Anchor is always in cluster 0
        self._unassigned = list(range(1, self._num_items))

        # Initialize the mean and confidence intervals
        self._means = np.zeros((self._num_items,))
        self._lower = np.ones((self._num_items,)) * -np.inf
        self._upper = np.ones((self._num_items,)) * np.inf

        # Initialize initial estimates of hat_p and hat_q
        self._init_hat_p = init_hat_p
        self._init_hat_q = init_hat_q
        self._hat_p = -np.inf if init_hat_p is None else init_hat_p
        self._hat_q = np.inf if init_hat_q is None else init_hat_q

    @abc.abstractmethod
    def calculate_conf_interval(self, mean: float) -> (float, float):
        """
        Calculates the confidence interval for an item with the specified empirical mean
        :param mean: Mean value of the item
        :return l: Lower limit of the confidence interval
        :return u: Upper limit of the confidence interval
        """
        raise NotImplementedError('Function calculate_conf_interval not implemented.')

    def select(self) -> list[tuple[int, int]]:
        """
        :return item_pairs: A list of the form [(item_i1, item_j1), (item_i2, item_j2), ...]
        """
        # Return all (anchor, item) pairs
        return [(self._anchor, item) for item in range(self._num_items) if item != self._anchor]

    def update(self, item_pairs: list[tuple[int, int]], observations: list[float]) -> None:
        """
        :param item_pairs: A list containing item pairs that were sampled
        :param observations: A list of same size as item_pairs containing the observation for each pair
        """
        # Update the time
        self._time += 1

        # Update means
        item_js = [item_j for (_, item_j) in item_pairs]
        obs = np.asarray(observations)
        self._means[item_js] = (self._means[item_js] * (self._time - 1) + obs) / self._time

        # Update the confidence intervals
        for item_j in item_js:
            l, u = self.calculate_conf_interval(self._means[item_j])
            self._lower[item_j] = max(self._lower[item_j], l)
            self._upper[item_j] = min(self._upper[item_j], u)

        # Calculate hat_p and hat_q
        self._hat_p = max(np.max(self._lower), self._hat_p)
        self._hat_q = min(np.min(self._upper), self._hat_q)

        # Assign items to clusters
        to_c1, to_c2 = [], []
        for item_j in self._unassigned:
            # Add item to the same cluster as the anchor
            if self._lower[item_j] > self._hat_q:
                self._clusters[item_j] = 0
                to_c1.append(item_j)

            # Add item to the other cluster
            if self._upper[item_j] < self._hat_p:
                self._clusters[item_j] = 1
                to_c2.append(item_j)

        # Update the set of unassigned items
        self._unassigned = [item for item in self._unassigned if item not in to_c1 + to_c2]

    def terminate(self) -> bool:
        """
        :return terminate: Is the algorithm ready to terminate?
        """
        return len(self._unassigned) == 0

    def progress(self) -> float:
        """
        :return progress: A measure of progress between 0 and 1
        """
        return 1 - (len(self._unassigned) + 1) / self._num_items

    def result(self) -> dict[str]:
        """
        :return result: A dictionary containing required output at the end of the algorithm
        """
        output = super().result()
        output['hat_p'] = self._hat_p
        output['hat_q'] = self._hat_q
        return output

    def config(self) -> dict:
        config = super().config()
        config['init_hat_p'] = self._init_hat_p
        config['init_hat_p'] = self._init_hat_q
        return config


class SplitItemsH(SplitItems):
    """
    Algorithm for partitioning items into two clusters using confidence intervals based on Hoeffding's lemma
    """

    def __init__(
            self,
            num_items: int,
            delta: float,
            init_hat_p: Optional[float] = None,
            init_hat_q: Optional[float] = None
    ):
        """
        :param num_items: Number of items to consider
        :param delta: Confidence parameter
        :param init_hat_p: An initial value of hat_p if available
        :param init_hat_q: An initial value of hat_q if available
        """
        super(SplitItemsH, self).__init__(
            num_items=num_items, delta=delta, init_hat_p=init_hat_p, init_hat_q=init_hat_q
        )

    def calculate_conf_interval(self, mean: float) -> (float, float):
        """
        :param mean: Mean value of the item
        :return l: Lower limit of the confidence interval
        :return u: Upper limit of the confidence interval
        """
        conf_int = np.sqrt(np.log(2 * (self._time + 1) * (self._time ** 2) * (self._num_items ** 2) / self._delta) /
                           (2 * self._time))
        return mean - conf_int, mean + conf_int

    def config(self) -> dict:
        config = super().config()
        config['name'] = 'SplitItemsH'
        return config


class SplitItemsS(SplitItems):
    """
    Algorithm for partitioning items into two clusters using SGLRT stopping rule
    """

    def __init__(
            self,
            num_items: int,
            delta: float,
            max_search_iters: int = 10,
            init_hat_p: Optional[float] = None,
            init_hat_q: Optional[float] = None
    ):
        """
        :param num_items: Number of items to consider
        :param delta: Confidence parameter
        :param max_search_iters: Maximum number of binary search refinements
        :param init_hat_p: An initial value of hat_p if available
        :param init_hat_q: An initial value of hat_q if available
        """
        super(SplitItemsS, self).__init__(
            num_items=num_items, delta=delta, init_hat_p=init_hat_p, init_hat_q=init_hat_q
        )
        self._max_search_iters = max_search_iters

    def calculate_conf_interval(self, mean: float) -> (float, float):
        """
        :param mean: Mean value of the item
        :return l: Lower limit of the confidence interval
        :return u: Upper limit of the confidence interval
        """
        # Return the trivial confidence interval if the mean is 0 or 1
        if mean == 0 or mean == 1:
            return 0.0, 1.0

        # Binary relative entropy function
        def _kl(p: float, q: float) -> float:
            return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

        # Find the lower confidence limit
        lower, higher, mid = 0, mean, mean / 2
        rhs = np.log(2 * self._time * (np.log(6 * self._time) ** 2) * (self._num_items ** 2) / self._delta)
        for _ in range(self._max_search_iters):
            z_val = self._time * _kl(mean, mid)
            if z_val < rhs:
                higher = mid
            elif z_val > rhs:
                lower = mid
            else:
                lower = mid
                break
            mid = (lower + higher) / 2
        lower_limit = lower

        # Find the upper confidence limit
        lower, higher, mid = mean, 1, (1 + mean) / 2
        for _ in range(self._max_search_iters):
            z_val = self._time * _kl(mean, mid)
            if z_val < rhs:
                lower = mid
            elif z_val > rhs:
                higher = mid
            else:
                higher = mid
                break
            mid = (lower + higher) / 2
        upper_limit = higher

        return lower_limit, upper_limit

    def config(self) -> dict:
        config = super().config()
        config['name'] = 'SplitItemsS'
        config['max_search_iters'] = self._max_search_iters
        return config


class QBCluster(Algorithm, abc.ABC):
    """
    Algorithm for partitioning items into k clusters
    """

    def __init__(
            self,
            num_items: int,
            delta: float,
            num_clusters: Optional[int] = None
    ):
        """
        :param num_items: Number of items to cluster
        :param delta: Confidence level
        :param num_clusters: Number of clusters to discover. If none, the optimal value will be automatically chosen
        """
        super(QBCluster, self).__init__(num_items=num_items, delta=delta, num_clusters=num_clusters)

        # Initialize the set of unclustered items
        self._unclustered = list(range(self._num_items))
        self._curr_cluster = 1

        # Initialize hat_p and hat_q
        self._hat_p = -np.inf
        self._hat_q = np.inf

        # Initialize the algorithm for dividing items into two clusters
        self._splitting_alg = self._get_splitting_alg()
        self._curr_cluster = 0

    def _get_splitting_alg(self) -> SplitItems:
        """
        Internal variant of get_splitting_alg
        """
        items = self._unclustered
        if self._num_clusters is None:
            init_hat_p = self._hat_p
            init_hat_q = self._hat_q
            delta = self._delta / (self._curr_cluster * (self._curr_cluster + 1))
        else:
            init_hat_p, init_hat_q = None, None
            delta = self._delta / (self._num_clusters - 1)

        return self.get_splitting_alg(items=items, delta=delta, init_hat_p=init_hat_p, init_hat_q=init_hat_q)

    @abc.abstractmethod
    def get_splitting_alg(
            self,
            items: list[int],
            delta: float,
            init_hat_p: Optional[float] = None,
            init_hat_q: Optional[float] = None
    ) -> SplitItems:
        """
        Returns an instance of the splitting algorithm for partitioning these items into two clusters
        :param items: Items to split in two clusters
        :param delta: Confidence parameter to use for splitting item into two clusters
        :param init_hat_p: Initial value of hat_p
        :param init_hat_q: Initial value of hat_q
        :return alg: An instance of SplitItems
        """
        raise NotImplementedError('Function get_splitting_alg not implemented.')

    def select(self) -> list[tuple[int, int]]:
        """
        :return item_pairs: A list of the form [(item_i1, item_j1), (item_i2, item_j2), ...]
        """
        # Select items using the splitting algorithm
        return [(self._unclustered[a], self._unclustered[j]) for (a, j) in self._splitting_alg.select()]

    def update(self, item_pairs: list[tuple[int, int]], observations: list[float]) -> None:
        """
        :param item_pairs: A list containing item pairs that were sampled
        :param observations: A list of same size as item_pairs containing the observation for each pair
        """
        # Update the splitting algorithm
        pairs = [(self._unclustered.index(a), self._unclustered.index(j)) for (a, j) in item_pairs]
        self._splitting_alg.update(item_pairs=pairs, observations=observations)

        # Reset the splitting algorithm if it has terminated
        if self._splitting_alg.terminate():

            # Isolate the pure and mixed cluster
            output = self._splitting_alg.result()
            partition = output['clusters']
            pure_cluster = [item for idx, item in enumerate(self._unclustered) if partition[idx] == 0]
            mixed_cluster = [item for idx, item in enumerate(self._unclustered) if partition[idx] == 1]

            # Update cluster memberships
            for item in pure_cluster:
                self._clusters[item] = self._curr_cluster
            self._curr_cluster += 1

            # Update unclustered items
            self._unclustered = mixed_cluster[:]

            # Update hat_p and hat_q
            self._hat_p = max(self._hat_p, output['hat_p'])
            self._hat_q = min(self._hat_q, output['hat_q'])

            # Reinitialize the splitting algorithm
            if len(self._unclustered) > 0:
                if self._num_clusters is not None and self._curr_cluster == self._num_clusters - 1:
                    # Assign remaining items to the last cluster
                    self._splitting_alg = None
                    for item in self._unclustered:
                        self._clusters[item] = self._curr_cluster
                    self._unclustered = []
                else:
                    self._splitting_alg = self._get_splitting_alg()
            else:
                self._splitting_alg = None

    def terminate(self) -> bool:
        """
        :return terminate: Is the algorithm ready to terminate?
        """
        return len(self._unclustered) == 0

    def progress(self) -> float:
        """
        :return progress: A measure of progress between 0 and 1
        """
        return 1 - (len(self._unclustered) + 1) / self._num_items

    def config(self) -> dict:
        config = super().config()
        return config


class QBClusterH(QBCluster):
    """
    Partitions items into k clusters by repeatedly calling SplitItemsH
    """

    def __init__(
            self,
            num_items: int,
            delta: float,
            num_clusters: Optional[int] = None
    ):
        """
        :param num_items: Number of items to cluster
        :param delta: Confidence level
        :param num_clusters: Number of clusters to discover. If none, the optimal value will be automatically chosen
        """
        super(QBClusterH, self).__init__(num_items=num_items, num_clusters=num_clusters, delta=delta)

    def get_splitting_alg(
            self,
            items: list[int],
            delta: float,
            init_hat_p: Optional[float] = None,
            init_hat_q: Optional[float] = None
    ) -> SplitItemsH:
        """
        Returns an instance of the splitting algorithm for partitioning these items into two clusters
        :param items: Items to split in two clusters
        :param delta: Confidence parameter to use for splitting item into two clusters
        :param init_hat_p: Initial value of hat_p
        :param init_hat_q: Initial value of hat_q
        :return alg: An instance of SplitItems
        """
        return SplitItemsH(num_items=len(items), delta=delta, init_hat_p=init_hat_p, init_hat_q=init_hat_q)

    def config(self) -> dict:
        config = super().config()
        config['name'] = 'QBClusterH'
        return config


class QBClusterS(QBCluster):
    """
    Partitions items into k clusters by repeatedly calling SplitItemsS
    """

    def __init__(
            self,
            num_items: int,
            delta: float,
            max_search_iters: int = 10,
            num_clusters: Optional[int] = None
    ):
        """
        :param num_items: Number of items to cluster
        :param delta: Confidence level
        :param max_search_iters: Parameter used in SplitItemsS
        :param num_clusters: Number of clusters to discover. If none, the optimal value will be automatically chosen
        """
        self._max_search_iters = max_search_iters
        super(QBClusterS, self).__init__(num_items=num_items, num_clusters=num_clusters, delta=delta)

    def get_splitting_alg(
            self,
            items: list[int],
            delta: float,
            init_hat_p: Optional[float] = None,
            init_hat_q: Optional[float] = None
    ) -> SplitItemsS:
        """
        Returns an instance of the splitting algorithm for partitioning these items into two clusters
        :param items: Items to split in two clusters
        :param delta: Confidence parameter to use for splitting item into two clusters
        :param init_hat_p: Initial value of hat_p
        :param init_hat_q: Initial value of hat_q
        :return alg: An instance of SplitItems
        """
        return SplitItemsS(
            num_items=len(items),
            delta=delta,
            init_hat_p=init_hat_p,
            init_hat_q=init_hat_q,
            max_search_iters=self._max_search_iters
        )

    def config(self) -> dict:
        config = super().config()
        config['name'] = 'QBClusterS'
        config['max_search_iters'] = self._max_search_iters
        return config
