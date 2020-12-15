from pomegranate import State
from .tables import PriorProbabilityTable, ConditionalProbabilityTable


class Node(State):

    def __init__(self, name, states, parents=None, npt=None, **kwargs):

        self.parents = parents
        self.states = states

        if self.prior():
            distribution = PriorProbabilityTable(
                label=name,
                states=self.states,
                values=npt
            )
        else:
            distribution = ConditionalProbabilityTable(
                label=name,
                states=self.states,
                parent_nodes=self.parents,
                values=npt
            )

        super().__init__(distribution, name)
        self.name = name

        for key in ['group', 'description', 'level']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    @property
    def states(self):
        return self.__states

    @states.setter
    def states(self, states):
        if (states is None) or (states == 'YN'):
            self.__states = ['No', 'Yes']
        elif states == 'PN':
            self.__states = ['Negative', 'Positive']
        elif states == 'TF':
            self.__states = ['False', 'True']
        else:
            self.__states = states

    def parent_sizes(self):
        return [len(parent) for parent in self.parents]

    def parent_names(self):
        return [parent.name for parent in self.parents]

    def prior(self):
        return self.parents is None

    @property
    def npt(self):
        return self.distribution

    @npt.setter
    def npt(self, values):
        self.distribution.values = values

    def __str__(self):
        return f"node('{self.name}')"

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.states)
