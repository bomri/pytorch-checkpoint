import unittest

from pytorchcheckpoint.checkpoint import CheckpointHandler


class TestGetStoredVariables(unittest.TestCase):

    def setUp(self):
        self.checkpoint_handler = CheckpointHandler()

    def test_get_var(self):
        self.checkpoint_handler.store_var(var_name='loss', iteration=0, value=1.0)
        self.checkpoint_handler.store_var(var_name='loss', iteration=1, value=0.9)
        self.checkpoint_handler.store_var(var_name='loss', iteration=2, value=0.8)

        a = self.checkpoint_handler.get_var(var_name='loss', iteration=0)
        b = self.checkpoint_handler.get_var(var_name='loss', iteration=1)
        c = self.checkpoint_handler.get_var(var_name='loss', iteration=2)
        d = self.checkpoint_handler.get_var(var_name='loss', iteration=3)
        e = self.checkpoint_handler.get_var(var_name='losses', iteration=0)

        self.assertEqual(a, 1.0)
        self.assertEqual(b, 0.9)
        self.assertEqual(c, 0.8)
        self.assertFalse(d)
        self.assertFalse(e)

    def test_get_var_with_header(self):
        self.checkpoint_handler.store_var_with_header(header='train', var_name='loss', iteration=0, value=1.0)
        self.checkpoint_handler.store_var_with_header(header='train', var_name='loss', iteration=1, value=0.9)

        a = self.checkpoint_handler.get_var_with_header(header='train', var_name='loss', iteration=0)
        b = self.checkpoint_handler.get_var_with_header(header='train', var_name='loss', iteration=1)
        c = self.checkpoint_handler.get_var_with_header(header='Train', var_name='loss', iteration=1)
        d = self.checkpoint_handler.get_var_with_header(header='train', var_name='losses', iteration=1)
        e = self.checkpoint_handler.get_var_with_header(header='train', var_name='loss', iteration=15)

        self.assertEqual(a, 1.0)
        self.assertEqual(b, 0.9)
        self.assertFalse(c)
        self.assertFalse(d)
        self.assertFalse(e)


if __name__ == '__main__':
    unittest.main()
