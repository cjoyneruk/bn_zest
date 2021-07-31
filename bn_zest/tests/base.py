import os
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
MODELS_DIR = os.path.join(FILE_DIR, 'models')

from bn_zest import BayesianNetwork, Node

def create_test_model():

    np.random.seed(666)

    a = Node('A', states=['No', 'Yes'], description='This is the prior node', group='inputs_1')
    b = Node('B', states=['Low', 'Med', 'High'], npt=[0.6, 0.15, 0.25], description='This is the output node', group='outputs_1')

    c = Node('C', states=['Positive', 'Negative'], parents=[a, b], npt='random', group='inputs_1')

    d = Node('D', states=['Red', 'Green', 'Blue'], parents=[c], npt='random', group='inputs_2')

    e = Node('E', states=['Down', 'Up'], parents=[c, d], group='outputs_2')

    f = Node('F', states=['Red', 'Green', 'Blue'], parents=[e], npt='random', group='outputs_2')

    return BayesianNetwork('Zest test network',
        id='zest_test_network',
        variables=[a, b, c, d, e, f],
        description='This is my BN',
        input_groups=[
            {'id': 'inputs_1', 'name': 'Inputs 1', 'description': 'Description for inputs 1'},
            {'id': 'inputs_2', 'name': 'Inputs 2', 'description': 'Description for inputs 2'}
        ],
        output_groups=[
            {'id': 'outputs_1', 'name': 'Outputs 1', 'description': 'Description for outputs 1'},
            {'id': 'outputs_2', 'name': 'Outputs 2', 'description': 'Description for outputs 2'}
        ],)


class ErrorTestMixin:

    @staticmethod
    def load_model(filename, **kwargs):
        file, extension = os.path.splitext(filename)
        return getattr(BayesianNetwork, f'from_{extension[1:]}')(os.path.join(MODELS_DIR, filename), **kwargs)

    @staticmethod
    def save_model(model, filename):
        file, extension = os.path.splitext(filename)
        return getattr(model, f'to_{extension[1:]}')(os.path.join(MODELS_DIR, filename))

    def assertRaisesWithMessage(self, excClass, callableObj, msg, *args, **kwargs):

        self.assertRaises(excClass, callableObj, *args, **kwargs)

        try:
            callableObj(*args, **kwargs)
        except excClass as error:
            self.assertEqual(str(error), msg)

    def assertListAlmostEqual(self, x, y, places, **kwargs):

        for a, b in zip(x, y):

            if (isinstance(a, list)):
                self.assertListAlmostEqual(a, b, places)
            else:
                self.assertAlmostEqual(a, b, places, **kwargs)