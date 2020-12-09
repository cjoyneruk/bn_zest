from pomegranate import State
from .tables import PriorProbabilityTable, ConditionalProbabilityTable


class Node(State):

    def __init__(self, name, states=None, parents=None, npt=None, **kwargs):

        # - Initialize state with empty distribution and add later during NPT setting
        super().__init__(None, name)

        self.name = name
        self.parents = []
        self.children = []

        if parents is not None:
            self.add_parents(parents)

        self.states = states

        for key in ['group', 'description', 'level', 'force_values']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

        self.distribution = npt

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

    def add_parents(self, nodes):

        for node in nodes:
            if node not in self.parents:
                self.parents.append(node)

            if self not in node.children:
                node.children.append(self)

    def add_children(self, nodes):

        for node in nodes:
            if node not in self.children:
                self.children.append(node)

            if self not in node.parents:
                node.parents.append(self)

    def parent_sizes(self):
        return [len(parent) for parent in self.parents]

    def parent_names(self):
        return [parent.name for parent in self.parents]

    def prior(self):
        return len(self.parents) == 0

    def _get_distribution_class(self):
        return PriorProbabilityTable if self.prior() else ConditionalProbabilityTable

    @property
    def distribution(self):
        return self.__distribution

    @distribution.setter
    def distribution(self, values):
        self.__distribution = self._get_distribution_class()(self, values)

    @property
    def npt(self):
        return self.distribution

    @npt.setter
    def npt(self, values):
        self.distribution = values

    def __str__(self):
        return f"node('{self.name}')"

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.states)
