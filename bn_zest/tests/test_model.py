import unittest
from . import base
from bn_zest import BayesianNetwork, Node


class TestModel(base.ErrorTestMixin, unittest.TestCase):
        
    def setUp(self):

        self.model = base.create_test_model()

    def test_model_instance(self):
        self.assertIsInstance(self.model, BayesianNetwork)

    def test_model_variable_ids(self):
        self.assertListEqual(self.model.variable_ids, ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_model_variable_names(self):
        self.assertListEqual(self.model.variable_names, ['A', 'B', 'C', 'D', 'E', 'F'])

    def test_get_variables(self):        
        for idx in self.model.variable_ids:
            self.assertIsInstance(self.model[idx], Node)

    def test_model_prediction(self):
        inputs = {
            'a': 'No',
            'c': 'Positive',
            'd': 'Red'
        }
        
        probs = self.model.predict_proba(X=inputs)
        
        outputs = {
            'b': [0.6, 0.15, 0.25],
            'e': [0.5, 0.5],
            'f': [0.22260035069952577, 0.3346594875701657, 0.44274016173030845]
        }

        for idx, values in probs.items():            
            self.assertListAlmostEqual(values, outputs[idx], places=6)

if __name__ == '__main__':
    unittest.main()