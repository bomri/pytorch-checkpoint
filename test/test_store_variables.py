import unittest

from pytorchcheckpoint.checkpoint import CheckpointHandler


class TestStoreVariables(unittest.TestCase):

    def setUp(self):
        self.checkpoint_handler = CheckpointHandler()

    def test_store_var(self):
        self.checkpoint_handler.store_var(var_name='loss', iteration=0, value=1.0)
        self.checkpoint_handler.store_var(var_name='loss', iteration=1, value=0.9)
        self.checkpoint_handler.store_var(var_name='loss', iteration=2, value=0.8)

        self.checkpoint_handler.store_var(var_name='top1', iteration=0, value=80)
        self.checkpoint_handler.store_var(var_name='top1', iteration=1, value=85)
        self.checkpoint_handler.store_var(var_name='top1', iteration=2, value=90)
        self.checkpoint_handler.store_var(var_name='top1', iteration=3, value=91)

        loss_output = {0: 1.0, 1: 0.9, 2: 0.8}
        top1_output = {0: 80, 1: 85, 2: 90, 3: 91}
        self.assertDictEqual(self.checkpoint_handler.loss, loss_output)
        self.assertDictEqual(self.checkpoint_handler.top1, top1_output)

    def test_store_var_with_header(self):
        self.checkpoint_handler.store_var_with_header(header='train', var_name='loss', iteration=0, value=1.0)
        self.checkpoint_handler.store_var_with_header(header='train', var_name='loss', iteration=0, value=1.0)
        self.checkpoint_handler.store_var_with_header(header='train', var_name='loss', iteration=1, value=0.9)
        self.checkpoint_handler.store_var_with_header(header='train', var_name='loss', iteration=2, value=0.7)

        self.checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=0, value=80)
        self.checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=1, value=85)
        self.checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=2, value=90)
        self.checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=3, value=91)

        output_train = {'loss': {0: 1.0, 1: 0.9, 2: 0.7}, 'top1': {0: 80, 1: 85, 2: 90, 3: 91}}
        self.assertDictEqual(self.checkpoint_handler.train, output_train)

        self.checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=0, value=70)
        self.checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=1, value=75)
        self.checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=2, value=80)
        self.checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=3, value=85)

        output_valid = {'top1': {0: 70, 1: 75, 2: 80, 3: 85}}
        self.assertDictEqual(self.checkpoint_handler.valid, output_valid)


if __name__ == '__main__':
    unittest.main()
