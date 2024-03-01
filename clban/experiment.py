import numpy as np
from tqdm import tqdm
from clban.oracle import Oracle
from clban.algorithms import Algorithm


class Experiment:

    def __init__(self, oracle: Oracle, alg: Algorithm, max_steps: int = 1000000):
        """
        :param oracle: An instance of the Oracle class
        :param alg: An instance of the Algorithm class
        :param max_steps: Maximum number of steps to take
        """
        self._oracle = oracle
        self._alg = alg
        self._max_steps = max_steps

    def run(self) -> (int, np.ndarray):
        """
        :return results: A dictionary containing logging information
        """
        # Run the algorithm
        time = 0
        samples = 0
        early_termination = False
        with tqdm(total=1) as pbar:
            while not self._alg.terminate():

                # Execute one step
                item_pairs = self._alg.select()
                observations = self._oracle.sample(item_pairs)
                self._alg.update(item_pairs, observations)
                pbar.update(self._alg.progress() - pbar.n)
                time += 1
                samples += len(item_pairs)

                # Stop early if maximum number of steps is reached
                if time == self._max_steps and not self._alg.terminate():
                    early_termination = True
                    break

        # Get the clusters
        clusters = None if early_termination else self._alg.result().get('clusters', None)
        return {'early_termination': early_termination, 'time': time, 'samples': samples, 'clusters': clusters}

    def config(self) -> dict:
        return {'max_steps': self._max_steps, 'oracle_config': self._oracle.config(), 'alg_config': self._alg.config()}
