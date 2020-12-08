import pandas as pd
import numpy as np
import pomegranate as pg

pd.set_option('expand_frame_repr', False)


class PriorProbabilityTable(pg.DiscreteDistribution):

    def __init__(self, target, values, force_values=False):

        self.target = target
        self.force_values = force_values
        self.values = values

        super().__init__(dict(zip(self.target.states, self.values)))

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, values):

        if isinstance(values, list):
            values = np.array(values)

        if not np.array_equal([len(self.target)], values.shape):
            raise ValueError(f"The distribution supplied for '{self.target.name}' is not the correct shape")

        if (not self.force_values) and (not all(p == 1 for p in self.values.sum(axis=0))):
            raise ValueError(f"The probabilities for '{self.target.name}' do not sum to 1")

        self.__values = values / values.sum(axis=0)

    def to_df(self):
        return pd.DataFrame(self.values, index=self.target.states, columns=[self.target.name])

    def __repr__(self):
        return str(self.to_df().round(3))


class ConditionalProbabilityTable(pg.ConditionalProbabilityTable):

    def __init__(self, target, values, force_values=False):

        self.target = target
        self.force_values = force_values
        self.values = values

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
            raise ValueError(f"The distribution supplied for '{self.target.name}' is should be of shape {shape}")

        if (not self.force_values) and (not all(p == 1 for p in self.values.sum(axis=0))):
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