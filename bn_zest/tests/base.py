import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
sys.path.append(ROOT_DIR)

class ErrorTestMixin:

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