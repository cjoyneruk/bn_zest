import numpy as np
from pomegranate import State
from .tables import ProbTable


class Node(State):

    def __init__(self, name, states=None, parents=None, npt=None, **kwargs):

        self.name = name
        self.parents = []
        self.children = []

        if parents is not None:
            self.add_parents(parents)

        self.states = states

        if 'group' in kwargs:
            self.group = kwargs['group']

        if 'description' in kwargs:
            self.description = kwargs['description']

        if 'level' in kwargs:
            self.level = kwargs['level']

        if npt is None:
            npt = np.ones([len(states)] + [len(p) for p in self.parents])

        self.NPT = npt
        super(Node, self).__init__(self.NPT.dist, name)


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

    def is_prior(self):
        return len(self.parents) == 0

    @property
    def NPT(self):
        return self.__NPT

    @NPT.setter
    def NPT(self, npt):
        self.__NPT = ProbTable(self, npt)

    def __str__(self):
        return f"node('{self.name}')"

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.states)