import pandas as pd
import numpy as np
from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable as BaseCPT

pd.set_option('expand_frame_repr', False)


class PriorProbabilityTable(DiscreteDistribution):

    def __new__(cls, *args):

        if isinstance(args[0], dict):
            return super(PriorProbabilityTable, cls).__new__(cls, args[0])

        target, values = args[:2]
        if not cls._check_values(values):
            raise ValueError('Probability values must sum to 1')

        return super(PriorProbabilityTable, cls).__new__(cls, dict(zip(target.states, values)))

    def __init__(self, *args):

        if not isinstance(args[0], dict):
            self.target = args[0]
            self.values = args[1]

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, values):

        if isinstance(values, list):
            values = np.array(values)

        if not np.array_equal([len(self.target)], values.shape):
            raise ValueError(f"The distribution supplied for '{self.target.name}' is not the correct shape")

        if values.sum(axis=0) != 1:
            raise ValueError(f"The probabilities for '{self.target.name}' do not sum to 1")

        self.__values = values / values.sum(axis=0)

    @staticmethod
    def _check_values(values):
        return sum(values) == 1

    def to_df(self):
        return pd.DataFrame(self.values, index=self.target.states, columns=[self.target.name])

    def __repr__(self):
        return str(self.to_df().round(3))

    def __str__(self):
        return str(self.to_df().round(3))


class ConditionalProbabilityTable(BaseCPT):

    def __init__(self, *args):

        if isinstance(args[0], np.ndarray):
            super().__init__(*args)

        else:

            self.target = args[0]
            self.values = args[1]

            super().__init__(self.input_values(), self.parent_distributions())

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, values):

        if isinstance(values, list):
            values = np.array(values)

        shape = [len(self.target)] + [len(parent) for parent in self.target.parents]

        if not np.array_equal(shape, values.shape):
            raise ValueError(f"The distribution supplied for '{self.target.name}' should be of shape {shape}")

        f = np.abs(values.sum(axis=0) - 1) > 1e-10
        if any(f.flatten()):
            raise ValueError(f"The probabilities for '{self.target.name}' do not sum to 1")

        self.__values = values / values.sum(axis=0)

    def to_df(self):

        values = self.values.reshape(len(self.target), np.prod(self.target.parent_sizes()))

        levels = (parent.states for parent in self.target.parents)
        headings = pd.MultiIndex.from_product(levels, names=self.target.parent_names())

        return pd.DataFrame(values, index=self.target.states, columns=headings)

    def parent_distributions(self):
        return [parent.distribution for parent in self.target.parents]

    def input_values(self):

        # - Rearrange to get npt
        input_npt = self.to_df().transpose()

        # - Rename parents to avoid potential conflict with state names
        new_parent_names = [f'parent_{name}' for name in self.target.parent_names()]
        input_npt.index.rename(new_parent_names, inplace=True)
        input_npt = input_npt.reset_index().melt(
            id_vars=new_parent_names,
            value_vars=self.target.states
        )
        return input_npt.values

    def __repr__(self):
        return str(self.to_df().round(3))

    def __str__(self):
        return str(self.to_df().round(3))
