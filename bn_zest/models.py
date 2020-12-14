import json
import os
import pomegranate
from .parsers import from_cmpx, to_cmpx
from .nodes import Node
import pandas as pd
import numpy as np


class BayesianNetwork(pomegranate.BayesianNetwork):

    def __init__(self, name=None, nodes=None, compiled=False, output_nodes=None):

        super().__init__(name)
        self.add_states(*nodes)

        for node in filter(lambda x: not x.prior(), self.nodes):
            for parent in node.parents:
                self.add_edge(parent, node)

        self.compiled = compiled
        if self.compiled:
            self.bake()

        self.output_nodes = output_nodes

    @property
    def nodes(self):
        return self.states

    @property
    def node_names(self):
        return [x.name for x in self.states]

    def predict_proba(self, X=None, max_iterations=100, check_input=True, n_jobs=1):

        """

        :X NoneType, dict, DataFrame:
        Either a dictionary or dataframe of input values

        :param args:
        :param kwargs:
        :return: Marginal probabilities of output nodes
        """

        if not self.compiled:
            self.bake()

        if isinstance(X, dict):

            if all(isinstance(node, Node) for node in X.keys()):
                X = {node.name: state for node, state in X.items()}

            elif not all(isinstance(node, str) for node in X.keys()):
                raise TypeError('The keys must either be all Nodes or names of Nodes')

            for name, state in X.items():
                if name not in self.node_names:
                    raise KeyError(f"The name '{name}' does not match any nodes in the model")
                if state not in self[name].states:
                    raise KeyError(f"The state '{state}' is not a state of {name}")

            probs = super(BayesianNetwork, self).predict_proba(X, max_iterations=max_iterations, check_input=check_input, n_jobs=n_jobs)
            outputs = (list(p.parameters[0].values()) for p in probs if not isinstance(p, str))
            output_names = (name for name in self.node_names if name not in X.keys())
            return dict(zip(output_names, outputs))

        elif isinstance(X, pd.DataFrame):

            # - Check entries
            for col in X.columns:
                if col not in self.node_names:
                    raise KeyError(f"The name {col} is not a recognized node")

                states = X[col].unique()
                for state in states:
                    if state not in self[col].states:
                        raise KeyError(f"The state '{state}' is not a state of {col}")

                output_names = [name for name in self.node_names if name not in X.columns]
                probs = X.apply(self._convert_probs(output_names, max_iterations=max_iterations, check_input=check_input, n_jobs=n_jobs), axis=1)
                return pd.json_normalize(probs.values)

        if X is not None:
            raise TypeError('X must be either a dictionary or a pandas DataFrame')

        probs = super(BayesianNetwork, self).predict_proba({})
        outputs = (list(p.parameters[0].values()) for p in probs)
        return dict(zip(self.node_names, outputs))

    def _convert_probs(self, output_names, **kwargs):

        def get_output(x):
            probs = super(BayesianNetwork, self).predict_proba(dict(x), **kwargs)
            outputs = [list(p.parameters[0].values()) for p in probs if not isinstance(p, str)]
            return dict(zip(output_names, outputs))

        return get_output

    def predict(self, X, *args, **kwargs):

        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')

        # - Construct apply function
        def get_prediction(node):
            def get_state(values):
                return node.states[np.argmax(values)]
            return get_state

        X_pred = self.predict_proba(X).copy()

        for col in X_pred.columns:
            X_pred[col] = X_pred[col].apply(get_prediction(self[col]))

        return X_pred

    def sample(self, *args, **kwargs):
        if not self.compiled:
            self.bake()

        values = super(BayesianNetwork, self).sample(*args, **kwargs)
        return pd.DataFrame(values, columns=self.node_names)

    def fit(self, X, y=None, **kwargs):

        if y is not None:
            X = pd.concat((X, y), axis=1)

        super(BayesianNetwork, self).fit(X, **kwargs)

    @classmethod
    def from_file(cls, filename, file_type=None, *args, **kwargs):

        file, extension = os.path.splitext(filename)
        extension = extension.split('.')[1]

        if file_type is None:
            if extension not in ['cmpx']:
                raise TypeError('Please supply a cmpx file')
            file_type = extension

        elif file_type not in ['cmpx']:
            raise TypeError('Please supply a cmpx file_type')

        with open(filename, 'r') as file:
            data_string = file.read()
            data = json.loads(data_string)

        return getattr(cls, f'from_{file_type}')(data, *args, **kwargs)

    @classmethod
    def from_cmpx(cls, data, network=0, remove_disconnected_nodes=True):
        name, nodes = from_cmpx(data, network=network, remove_disconnected_nodes=remove_disconnected_nodes)
        return cls(name, nodes)

    def to_file(self, filename, file_type=None):

        file, extension = os.path.splitext(filename)
        extension = extension.split('.')[1]

        if file_type is None:
            if extension not in ['cmpx']:
                raise TypeError('Please supply a cmpx file')
            file_type = extension

        elif file_type not in ['cmpx']:
            raise TypeError('Please supply a cmpx file_type')

        data = getattr(self, f'to_{file_type}')()

        with open(filename, 'w') as file:
            jstring = json.dumps(data)
            file.write(jstring)

    def to_cmpx(self):
        return to_cmpx(self)

    def __getitem__(self, item):
        return self.states[self.node_names.index(item)]

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f'BayesianNetwork({self.name})'
