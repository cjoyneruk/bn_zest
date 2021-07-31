import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(ROOT_DIR)

import unittest
from bn_zest import BayesianNetwork
from models.construct_model import create_test_model

CMPX_DIR = os.path.join(FILE_DIR, 'models_cmpx')
JSON_DIR = os.path.join(FILE_DIR, 'models_json')

class ErrorTestMixin:

    def assertRaisesWithMessage(self, excClass, callableObj, msg, *args, **kwargs):

        self.assertRaises(excClass, callableObj, *args, **kwargs)

        try:
            callableObj(*args, **kwargs)
        except excClass as error:
            self.assertEqual(str(error), msg)


class TestCMPXParser(ErrorTestMixin, unittest.TestCase):

    def assertIsBayesianNetworkFromCMPX(self, filepath, **kwargs):
        model = BayesianNetwork.from_cmpx(filepath, **kwargs)
        self.assertIsInstance(model, BayesianNetwork)

    def test_cmpx_bendi_bn(self):
        self.assertIsBayesianNetworkFromCMPX(os.path.join(CMPX_DIR, 'bendi_bn_test.cmpx'))

    def test_cmpx_limbmodel_fail(self):

        model_path = os.path.join(CMPX_DIR, 'limbmodel_test.cmpx')

        self.assertRaisesWithMessage(
            ValueError,
            BayesianNetwork.from_cmpx,
            'The probabilities for Duration of Ischaemia do not sum to 1. Please change or use force_summation=True when reading from cmpx file',
            model_path
        )
    
    def test_cmpx_limbmodel_success(self):
        self.assertIsBayesianNetworkFromCMPX(os.path.join(CMPX_DIR, 'limbmodel_test.cmpx'), force_summation=True)

    def test_model_cmpx_output(self):
        
        model = create_test_model()

        output_path = os.path.join(CMPX_DIR, 'test_output_model.cmpx')

        model.to_cmpx(output_path)
        self.assertIsBayesianNetworkFromCMPX(output_path)


class TestJSONParser(ErrorTestMixin, unittest.TestCase):

    def assertIsBayesianNetworkFromJSON(self, filepath, **kwargs):
        model = BayesianNetwork.from_json(filepath, **kwargs)
        self.assertIsInstance(model, BayesianNetwork)

    def test_model_json_output(self):
        
        model = create_test_model()
        output_path = os.path.join(JSON_DIR, 'test_output_model.json')

        model.to_json(output_path)
        self.assertIsBayesianNetworkFromJSON(output_path)

    def test_model_json_fail(self):

        model_path = os.path.join(JSON_DIR, 'test_input_model.json')

        self.assertRaisesWithMessage(
            ValueError,
            BayesianNetwork.from_json,
            "The probabilities for 'A' do not sum to 1",
            model_path
        )

if __name__ == '__main__':
    unittest.main()