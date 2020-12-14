from pomegranate import State
from .tables import PriorProbabilityTable, ConditionalProbabilityTable


class Node(State):

    def __init__(self, name, states, parents=None, npt=None, **kwargs):

        # - Initialize state with empty distribution and add later during NPT setting
        self.parents = parents
        self.states = states
        super().__init__(None, name)
        self.npt = npt

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

    def _get_distribution_class(self):
        return PriorProbabilityTable if self.prior() else ConditionalProbabilityTable

    def get_distribution_kwargs(self, values):
        distribution_kwargs = {
            'label': self.name,
            'states': self.states,
            'values': values
        }

        if not self.prior():
            distribution_kwargs['parent_nodes'] = self.parents

        return distribution_kwargs

    @property
    def npt(self):
        return self.distribution

    @npt.setter
    def npt(self, values):
        self.distribution = self._get_distribution_class()(**self.get_distribution_kwargs(values))

    def __str__(self):
        return f"node('{self.name}')"

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.states)
