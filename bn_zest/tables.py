import pandas as pd
import numpy as np
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable

pd.set_option('expand_frame_repr', False)


class ProbTable:

    def __init__(self, target, values):

        if isinstance(values, list):
            values = np.array(values)

        shape = [len(target)] + target.parent_sizes()
        self.values = values/values.sum(axis=0)

        if not np.array_equal(shape, values.shape):
            raise ValueError(f"The NPT supplied for '{target.name}' is not the correct shape")

        self.target = target
        self.name = target.name
        self.count = 0
        self.dist = self.get_distribution()

    def to_df(self):

        if len(self.target.parents) == 0:

            return pd.DataFrame(self.values, index=self.target.states, columns=[self.target.name])

        else:

            values = self.values.reshape(len(self.target), np.prod(self.target.parent_sizes()))
            # - create NPT headings
            levels = []
            names = []
            for node in self.target.parents:
                levels.append(node.states)
                names.append(node.name)

            headings = pd.MultiIndex.from_product(levels, names=names)

            return pd.DataFrame(values, index=self.target.states, columns=headings)

    def is_prior(self):
        return self.target.is_prior()

    def get_distribution(self):
        if self.is_prior():
            return DiscreteDistribution(dict(zip(self.target.states, self.values)))
        else:
            # - Get distributions
            dists = [parent.NPT.dist for parent in self.target.parents]

            # - Rearrange to get npt
            input_npt = self.to_df().transpose()

            # - Rename parents to avoid potential conflict with state names
            new_parent_names = [f'parent_{name}' for name in self.target.parent_names()]
            input_npt.index.rename(new_parent_names, inplace=True)
            input_npt = input_npt.reset_index().melt(
                id_vars=new_parent_names,
                value_vars=self.target.states
            )
            return ConditionalProbabilityTable(input_npt.values, dists)

    def __repr__(self):
        return str(self.to_df().round(3))