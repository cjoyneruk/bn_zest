import json
import os
import re
import pomegranate
from .parsers import from_cmpx, to_cmpx, from_dict
from .nodes import Node
import pandas as pd
import numpy as np


class BayesianNetwork(pomegranate.BayesianNetwork):

    def __init__(self, name, description=None, variables=None, **kwargs):

        super().__init__(name)
        self.add_states(*variables)

        if description is not None:
            self.description = description

        for variable in filter(lambda x: not x.prior(), self.variables):
            for parent in variable.parents:
                self.add_edge(parent, variable)

        # Set id
        if 'id' not in kwargs:

            value = re.sub(r'[^a-z0-9]', '', self.name.lower())[:20]

            if len(value) < 5:
                value = value + '_network'

            self.id = value

        # Set remaining kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        if not re.match(r'^[A-Za-z0-9_]{1,20}$', value):
            raise ValueError(f'The id {value} must be at most 20 characters and contain only letters numbers or underscores')

        self.__id = value

    @property
    def variables(self):
        return self.states

    @property
    def variable_names(self):
        return [x.name for x in self.states]

    def _get_dict_proba(self, X, output_variables, check_states=True, **kwargs):

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
            for i, node in output_variables}

        return output

    def _get_DataFrame_proba(self, X, output_variables, **kwargs):

        # - Check states
        for name in X.columns:
            for state in X[name].unique():
                if (state not in self[name].states) and (state is not None):
                    raise ValueError(f"The state '{state}' is not a state of {name}")

        return pd.json_normalize(X.apply(lambda x: self._get_dict_proba(dict(x), output_variables, check_states=False, **kwargs), axis=1))

    def predict_proba(self, X=None, **kwargs):

        """

        :X NoneType, dict, DataFrame:
        Either a dictionary or dataframe of input values

        :param args:
        :param kwargs: See
        :return: Marginal probabilities of output variables
        """

        # - Check types
        if X is None:
            X = {}

        if (not isinstance(X, dict)) and (not isinstance(X, pd.DataFrame)):
            raise TypeError('X must be either a dictionary of pandas DataFrame')

        # - Check input names
        for name in list(X.keys()):
            if name not in self.variable_names:
                raise KeyError(f'The node {name} does not match any contained in the model')

        output_variables = [(self.variables.index(node), node) for node in self.variables if node.name not in list(X.keys())]

        self.bake()

        return getattr(self, f'_get_{type(X).__name__}_proba')(X, output_variables)

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
        return pd.DataFrame(values, columns=self.variable_names)

    def fit(self, X, y=None, **kwargs):

        if y is not None:
            X = pd.concat((X, y), axis=1)

        super(BayesianNetwork, self).fit(X, **kwargs)

    @classmethod
    def from_cmpx(cls, filename, network=0, **kwargs):

        with open(filename, 'r') as file:
            data_string = file.read()

        return cls(**from_cmpx(json.loads(data_string), network=network, **kwargs))

    @classmethod
    def from_dict(cls, data):        
        return cls(**from_dict(data))

    @classmethod
    def from_json(cls, filename):
        
        with open(filename, 'r') as file:
            data_string = file.read()
           
        return cls.from_dict(json.loads(data_string))

    def to_file(self, filename, file_type=None):

        file_types = ['cmpx', 'json']

        file, extension = os.path.splitext(filename)
        extension = extension.split('.')[1]

        if file_type is None:
            if extension not in file_types:
                raise TypeError(f'Please supply a {file_types} file')
            file_type = extension

        elif file_type not in file_types:
            raise TypeError(f'Please supply a {file_types} file_type')

        data = getattr(self, f'to_{file_type}')()

        with open(filename, 'w') as file:
            jstring = json.dumps(data)
            file.write(jstring)

    def to_cmpx(self):
        return to_cmpx(self)

    def to_dict(self):
        data = {
            'id': self.id,
            'name': self.name,            
        }

        for key in ['description', 'input_groups', 'output_groups']:
            if hasattr(self, key):
                data[key] = getattr(self, key)
        
        data['variables'] = [variable.to_dict() for variable in self.variables]        
        return data

    def to_json(self, filename=None):

        json_string = json.dumps(self.to_dict(), indent=2)

        if filename is None:
            return json_string
        else:
            with open(filename, 'w') as file:
                file.write(json_string)

    def __getitem__(self, item):
        return self.states[self.variable_names.index(item)]

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f'BayesianNetwork({self.name})'
