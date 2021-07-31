import os
import unittest
from . import base
from bn_zest import BayesianNetwork


class TestCMPXParser(base.ErrorTestMixin, unittest.TestCase):

    def test_cmpx_bendi_bn(self):
        model = self.load_model('bendi_bn_test.cmpx')
        self.assertIsInstance(model, BayesianNetwork)

    def test_cmpx_limbmodel_fail(self):

        model_path = os.path.join(base.MODELS_DIR, 'limbmodel_test.cmpx')

        self.assertRaisesWithMessage(
            ValueError,
            BayesianNetwork.from_cmpx,
            'The probabilities for treatment do not sum to 1. Please change or use force_summation=True when reading from file',
            model_path
        )
    
    def test_cmpx_limbmodel_success(self):
        model = self.load_model('limbmodel_test.cmpx', force_summation=True)
        self.assertIsInstance(model, BayesianNetwork)

    def test_model_cmpx_output(self):
        
        model = base.create_test_model()

        self.save_model(model, 'test_output_model.cmpx')
        model = self.load_model('test_output_model.cmpx')
        self.assertIsInstance(model, BayesianNetwork)
        


class TestJSONParser(base.ErrorTestMixin, unittest.TestCase):


    def test_model_json_output(self):
        
        model = base.create_test_model()
        self.save_model(model, 'test_output_model.json')
        model = self.load_model('test_output_model.json')
        self.assertIsInstance(model, BayesianNetwork)
        

    def test_model_json_fail(self):

        model_path = os.path.join(base.MODELS_DIR, 'test_input_model.json')

        self.assertRaisesWithMessage(
            ValueError,
            BayesianNetwork.from_json,
            "The probabilities for a do not sum to 1. Please change or use force_summation=True when reading from file",
            model_path
        )
