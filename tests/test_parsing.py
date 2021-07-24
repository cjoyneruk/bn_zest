import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(ROOT_DIR)

import unittest
from bn_zest import BayesianNetwork
from models.test_model import create_test_model

CMPX_DIR = os.path.join(FILE_DIR, 'models_cmpx')
JSON_DIR = os.path.join(FILE_DIR, 'models_json')

class TestParser(unittest.TestCase):

    def assertRaisesWithMessage(self, excClass, callableObj, msg, *args, **kwargs):

        self.assertRaises(excClass, callableObj, *args, **kwargs)

        try:
            callableObj(*args, **kwargs)
        except excClass as error:
            self.assertEqual(str(error), msg)

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



if __name__ == '__main__':
    unittest.main()