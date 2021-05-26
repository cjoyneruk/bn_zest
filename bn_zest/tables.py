import pandas as pd
import numpy as np
from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable as BaseCPT
import itertools

pd.set_option('expand_frame_repr', False)


class PriorProbabilityTable(DiscreteDistribution):

    def __new__(cls, label, states, values):
        cls._check_values(label, states, values)
        return super(PriorProbabilityTable, cls).__new__(cls, dict(zip(states, values)))

    def __init__(self, label, states, values):
        self.label = label
        self.states = states

    @property
    def values(self):
        return self.parameters[0].values()

    @values.setter
    def values(self, values):
        self._check_values(self.label, self.states, values)
        self.parameters = [dict(zip(self.states, values))]

    @staticmethod
    def _check_values(label, states, values):

        if len(states) != len(values):
            raise ValueError(f"The distribution supplied for '{label}' is not the correct shape")

        if abs(sum(values) - 1) > 1e-10:
            raise ValueError(f"The probabilities for '{label}' do not sum to 1")

    def to_df(self):
        return pd.DataFrame(self.values, index=self.states, columns=[self.label])

    def get_params(self, *args, **kwargs):
        params = {}
        for key in ['label', 'states', 'values']:
            params[key] = getattr(self, key)
        return params

    def copy(self):
        return self.__class__(**self.get_params())

    def __repr__(self):
        return str(self.to_df().round(3))

    def __str__(self):
        return str(self.to_df().round(3))


class ConditionalProbabilityTable(BaseCPT):

    def __init__(self, label, states, parent_nodes, values):
        self.label = label
        self.states = states
        self.parent_nodes = parent_nodes
        self.npt_shape = [len(states)] + [len(node) for node in self.parent_nodes]

        if isinstance(values, list):
            values = np.array(values)

        self._check_values(values)
        params = self._values_to_parameters(values)
        super().__init__(params, [p.distribution for p in self.parent_nodes])

    def parent_labels(self):
        return [parent.label for parent in self.parents]

    def state_list(self):
        return [parent.states for parent in self.parent_nodes] + [self.states]

    @property
    def values(self):
        params = np.array(self.parameters[0])[:, -1]
        shape = [len(parent) for parent in self.parents] + [len(self)]
        params = params.astype(float).reshape(shape)
        return np.moveaxis(params, -1, 0)

    @values.setter
    def values(self, values):

        if isinstance(values, list):
            values = np.array(values)

        self.parameters[0] = self._values_to_parameters(values)

    def _check_values(self, values):

        if not np.array_equal(self.npt_shape, values.shape):
            raise ValueError(f"The distribution supplied for '{self.label}' should be of shape {self.npt_shape}")

        f = np.abs(values.sum(axis=0) - 1) > 1e-10
        if any(f.flatten()):
            raise ValueError(f"The probabilities for '{self.label}' do not sum to 1")

    def _values_to_parameters(self, values):
        values = np.moveaxis(values, 0, -1).flatten()
        state_combs = list(itertools.product(*self.state_list()))
        return [list(states) + [float(value)] for states, value in zip(state_combs, values)]

    def to_df(self):

        values = self.values.reshape(self.npt_shape[0], np.prod(self.npt_shape[1:]))

        levels = (parent.states for parent in self.parent_nodes)
        headings = pd.MultiIndex.from_product(levels, names=self.parent_labels())

        return pd.DataFrame(values, index=self.states, columns=headings)

    def get_params(self, *args, **kwargs):
        params = {}
        for key in ['label', 'states', 'parent_nodes', 'values']:
            params[key] = getattr(self, key)
        return params

    def copy(self):
        return self.__class__(**self.get_params())

    def __repr__(self):
        return str(self.to_df().round(3))

    def __str__(self):
        return str(self.to_df().round(3))
