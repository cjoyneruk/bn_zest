from pomegranate import State
import re
import json
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

        if 'id' not in kwargs:
            self.id = re.sub(r'[^a-z0-9]', '', self.name.lower())[:20]

        for key in ['id', 'group', 'description']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        if not re.match(r'[A-Za-z0-9_]{1,20}$', value):
            raise ValueError(f'The id {value} must be at most 20 characters and contain only letters numbers or underscores')

        self.__id = value

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

    def to_dict(self):
        
        data = {
            'id': self.id,
            'name': self.name,
            'states': self.states
        }

        if self.parents is not None:
            data['parents'] = [parent.id for parent in self.parents]
        
        data['npt'] = self.npt.to_df().values.tolist()
        
        for key in ['description', 'group']:
            if hasattr(self, key):
                data[key] = getattr(self, key)
        
        return data

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self):
        return f"Node('{self.name}')"

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.states)

