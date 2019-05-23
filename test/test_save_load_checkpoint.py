import unittest
import torchvision.models as models
from torch import optim

from pytorchcheckpoint.checkpoint import CheckpointHandler


class TestLoadSaveCheckpoint(unittest.TestCase):

    def setUp(self):
        self.checkpoint_handler = CheckpointHandler()

    def test_generate_checkpoint_path(self):
        path2save = '/tmp'
        checkpoint_path = self.checkpoint_handler.generate_checkpoint_path(path2save=path2save)
        ext = ".pth.tar"
        self.assertTrue(checkpoint_path.endswith(ext))

    def test_save_checkpoint(self):
        model = models.resnet18()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        path2save = '/tmp'
        checkpoint_path = self.checkpoint_handler.generate_checkpoint_path(path2save=path2save)
        self.checkpoint_handler.save_checkpoint(checkpoint_path=checkpoint_path,
                                                iteration=25,
                                                model=model,
                                                optimizer=optimizer)

    def test_load_model(self):
        model = models.resnet18()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        path2save = '/tmp'
        checkpoint_path = self.checkpoint_handler.generate_checkpoint_path(path2save=path2save)
        self.checkpoint_handler.save_checkpoint(checkpoint_path=checkpoint_path,
                                                iteration=25,
                                                model=model,
                                                optimizer=optimizer)

        self.checkpoint_handler_new = self.checkpoint_handler.load_checkpoint(checkpoint_path)
        self.assertEqual(self.checkpoint_handler.optimizer_state_dict, self.checkpoint_handler_new.optimizer_state_dict)


if __name__ == '__main__':
    unittest.main()
