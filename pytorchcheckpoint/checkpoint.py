import os
import datetime
import torch


class CheckpointHandler:

    def __init__(self):
        self.prefix_name = 'checkpoint'

    def store_var(self, var_name, iteration, value):
        if hasattr(self, var_name):
            cur = getattr(self, var_name)
            cur[iteration] = value
            setattr(self, var_name, cur)
        else:
            setattr(self, var_name, {iteration: value})

    def store_var_with_header(self, header, var_name, iteration, value):
        if hasattr(self, header):
            cur_header = getattr(self, header)
            if var_name in cur_header:
                cur_header[var_name][iteration] = value
            else:
                cur_header[var_name] = {iteration: value}
            setattr(self, header, cur_header)
        else:
            setattr(self, header, {var_name: {iteration: value}})

    def generate_checkpoint_path(self, path2save):
        now = datetime.datetime.now()
        filename = self.prefix_name + '_' + now.strftime("D%d_%m_%Y_T%H_%M") + ".pth.tar"
        checkpoint_path = os.path.join(path2save, filename)
        return checkpoint_path

    def save_checkpoint(self, checkpoint_path, iteration, model, optimizer):
        self.model_state_dict = model.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()
        self.iteration = iteration

        torch.save(self, checkpoint_path)

    @staticmethod
    def load_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
