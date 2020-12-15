import json
import os
import pomegranate
from .parsers import from_cmpx, to_cmpx
from .nodes import Node
import pandas as pd
import numpy as np


class BayesianNetwork(pomegranate.BayesianNetwork):

    def __init__(self, name=None, nodes=None):

        super().__init__(name)
        self.add_states(*nodes)

        for node in filter(lambda x: not x.prior(), self.nodes):
            for parent in node.parents:
                self.add_edge(parent, node)

    @property
    def nodes(self):
        return self.states

    @property
    def node_names(self):
        return [x.name for x in self.states]

    def _get_dict_proba(self, X, output_nodes, check_states=True, **kwargs):

        # - Remove NoneTypes
        X = {key: value for key, value in X.items() if value is not None}

        # - Check states
        if check_states:
            for name, state in X.items():
                if state not in self[name].states:
                    raise ValueError(f"The state '{state}' is not a state of {name}")

        prob = super(BayesianNetwork, self).predict_proba(X, **kwargs)
        output = {
            node.name: [prob[i].parameters[0][state] for state in node.states]
            for i, node in output_nodes}

        return output

    def _get_DataFrame_proba(self, X, output_nodes, **kwargs):

        # - Check states
        for name in X.columns:
            for state in X[name].unique():
                if (state not in self[name].states) and (state is not None):
                    raise ValueError(f"The state '{state}' is not a state of {name}")

        return pd.json_normalize(X.apply(lambda x: self._get_dict_proba(dict(x), output_nodes, check_states=False, **kwargs), axis=1))

    def predict_proba(self, X=None, **kwargs):

        """

        :X NoneType, dict, DataFrame:
        Either a dictionary or dataframe of input values

        :param args:
        :param kwargs: See
        :return: Marginal probabilities of output nodes
        """

        # - Check types
        if X is None:
            X = {}

        if (not isinstance(X, dict)) and (not isinstance(X, pd.DataFrame)):
            raise TypeError('X must be either a dictionary of pandas DataFrame')

        # - Check input names
        for name in list(X.keys()):
            if name not in self.node_names:
                raise KeyError(f'The node {name} does not match any contained in the model')

        output_nodes = [(self.nodes.index(node), node) for node in self.nodes if node.name not in list(X.keys())]

        self.bake()

        return getattr(self, f'_get_{type(X).__name__}_proba')(X, output_nodes)

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

        supported_file_types = ['cmpx']

        if file_type is None:
            if extension not in supported_file_types:
                raise TypeError(f'Only file types of the form {supported_file_types} are supported')
            file_type = extension

        elif file_type not in supported_file_types:
            raise TypeError(f'Only file types of the form {supported_file_types} are supported')

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
