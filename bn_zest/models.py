import json
import os
import pomegranate
from .parsers import from_cmpx, to_cmpx
from .nodes import Node
import pandas as pd

class BayesianNetwork(pomegranate.BayesianNetwork):

    def __init__(self, name, nodes, compiled=False):

        super().__init__(name)
        self.add_states(*nodes)

        for node in self.nodes:
            for parent in node.parents:
                self.add_edge(parent, node)

        self.compiled = compiled
        if self.compiled:
            self.bake()

    @property
    def nodes(self):
        return self.states

    @property
    def node_names(self):
        return [x.name for x in self.states]

    def predict_proba(self, X=None, *args, **kwargs):

        """
        Aim - to implement with sklearn
        :X dict:
        :param args:
        :param kwargs:
        :return:
        """

        if not self.compiled:
            self.bake()

        evidence = {}
        if X is not None:

            if not isinstance(X, dict):
                raise TypeError('You must supply a dictionary of nodes and state values')

            if all(isinstance(node, str) for node in X.keys()):
                evidence = {node: state for node, state in X.items()}

            elif all(isinstance(node, Node) for node in X.keys()):
                evidence = {node.name: state for node, state in X.items()}

            else:
                raise TypeError('The keys must either be all Nodes or names of Nodes')

        probs = super(BayesianNetwork, self).predict_proba(evidence)
        probs = [list(p.parameters[0].values()) for p in probs if not isinstance(p, str)]
        output_nodes = [node.name for node in self.nodes if node not in evidence.keys()]
        return dict(zip(output_nodes, probs))

    def sample(self, *args, **kwargs):
        if not self.compiled:
            self.bake()

        values = super(BayesianNetwork, self).sample(*args, **kwargs)
        return pd.DataFrame(values, columns=self.node_names)


    @classmethod
    def from_file(cls, filename, file_type=None, *args, **kwargs):

        file, extension = os.path.splitext(filename)
        extension = extension.split('.')[1]

        if file_type is None:
            if extension not in ['cmpx']:
                raise TypeError('Please supply a cmpx file')
            file_type = extension

        else:
            if file_type not in ['cmpx']:
                raise TypeError('Please supply a cmpx file_type')

        with open(filename, 'r') as file:
            data_string = file.read()
            data = json.loads(data_string)

        return getattr(cls, f'from_{file_type}')(data, *args, **kwargs)

    @classmethod
    def from_cmpx(cls, data, network=0, remove_disconnected_nodes=True):
        name, nodes = from_cmpx(data, network=network, remove_disconnected_nodes=remove_disconnected_nodes)
        return cls(name, nodes)

    def to_file(self, filename, file_type=None):

        file, extension = os.path.splitext(filename)
        extension = extension.split('.')[1]

        if file_type is None:
            if extension not in ['cmpx']:
                raise TypeError('Please supply a cmpx file')
            file_type = extension

        else:
            if file_type not in ['cmpx']:
                raise TypeError('Please supply a cmpx file_type')

        data = getattr(self, f'to_{file_type}')()

        with open(filename, 'w') as file:
            jstring = json.dumps(data)
            file.write(jstring)

    def to_cmpx(self):
        return to_cmpx(self)

    def __getitem__(self, item):
        return self.states[self.node_names.index(item)]

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f'BayesianNetwork({self.name})'
