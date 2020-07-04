from tensorboardX import SummaryWriter
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, log_dir):
        # create the summary writer object
        self._writer = SummaryWriter(log_dir)

        self.global_step_train = 0
        self.global_step_valid = 0
        self.global_step_test = 0

    def __del__(self):
        self._writer.close()

    def add_scalar(self, name, scalar_value, flag):
        assert isinstance(scalar_value, float), type(scalar_value)
        if flag == 'train':
            step = self.global_step_train
        elif flag == 'valid':
            step = self.global_step_valid
        elif flag == 'test':
            step = self.global_step_test
        self._writer.add_scalar(name, scalar_value, step)

    def add_image(self, name, img_tensor, flag):
        assert isinstance(img_tensor, torch.Tensor), type(img_tensor)
        if flag == 'train':
            step = self.global_step_train
        elif flag == 'valid':
            step = self.global_step_valid
        elif flag == 'test':
            step = self.global_step_test
        self._writer.add_image(name, img_tensor, step)

    def step_train(self):
        self.global_step_train += 1

    def step_valid(self):
        self.global_step_valid += 1

    def step_test(self):
        self.global_step_test += 1

