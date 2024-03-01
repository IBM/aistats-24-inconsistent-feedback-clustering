Python code implementing the algorithms presented in our AISTATS-24 paper titled "Clustering Items From Adaptively Collected Inconsistent Feedback." These algorithms cluster a set of items using pairwise similarity feedback from an oracle.

# Installing the required packages

Run `poetry install`. This installs `numpy`, `scipy`, `scikit-learn`, and `tqdm`. Alternatively, you can install these packages manually.

# Running the code

1. Create a configuration file like `example_config.json`
2. Run `poetry run python run.py path/to/config.json`

The output, consisting of key-value pairs, will be stored as a `*.pkl` file that can be read using the `pickle` package. The keys are named so that their purpose is self-explanatory. To load the output file, use something like:

```python
import pickle

output = pickle.load(open("output.pkl", "rb"))
```

# Code structure

```
|-- clban
|     |-- algorithms.py    # Implements the algorithms
|     |-- oracle.py        # Implements the oracle
|     |-- experiment.py    # Implements the experimental setup
|     |-- utils.py         # Utility functions for evaluation
|
|-- run.py                 # For running a single configuration
|-- example_config.json    # An example configuration file
```
