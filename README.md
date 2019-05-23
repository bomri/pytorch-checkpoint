# pytorch-checkpoint

[![PyPI version](https://badge.fury.io/py/pytorchcheckpoint.svg)](https://badge.fury.io/py/pytorchcheckpoint)


This package supports saving and loading PyTorch training checkpoints. It is useful when trying the resume model training from a previous step, and can become handy when working with spot instances or when trying to reproduce results.

A model is saved not only with its weights, as one might do for later inference, but the entire state of the model, including the optimizer state and parameters.

In addition, it allows saving metrics and other values generated while training, such as accuracy and loss values. This makes it possible to recreate the learning curves from past values and continue to update them as training proceed.


------------------


## Prerequisites
Developed with **Python 3.7.3**, but should be compatible with previous Python version.
```
torch==1.1.0
torchvision==0.3.0
```

## Installation
```pip install pytorchcheckpoint```

## Usage
```python
from pytorchcheckpoint.checkpoint import CheckpointHandler
checkpoint_handler = CheckpointHandler()
```

#### Storing values and metrics for each epoch/iteration. For example, the loss value: 
```python
checkpoint_handler.store_var(var_name='loss', iteration=0, value=1.0)
checkpoint_handler.store_var(var_name='loss', iteration=1, value=0.9)
checkpoint_handler.store_var(var_name='loss', iteration=2, value=0.8)
```

#### Storing values and metrics per set: train/valid/test for each epoch/iteration. For example, the top1 value of the train and valid sets: 
```python
checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=0, value=80)
checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=1, value=85)
checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=2, value=90)
checkpoint_handler.store_var_with_header(header='train', var_name='top1', iteration=3, value=91)

checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=0, value=70)
checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=1, value=75)
checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=2, value=80)
checkpoint_handler.store_var_with_header(header='valid', var_name='top1', iteration=3, value=85)
```

#### Save checkpoint:
```python
import torchvision.models as models
from torch import optim
checkpoint_handler.store_var(var_name='loss', iteration=0, value=1.0)
model = models.resnet18()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
path2save = '/tmp'
checkpoint_path = checkpoint_handler.generate_checkpoint_path(path2save=path2save)
checkpoint_handler.save_checkpoint(checkpoint_path=checkpoint_path, iteration=25, model=model, optimizer=optimizer)
```

#### Load checkpoint:
```python
checkpoint_path = '<checkpoint_path>'
checkpoint_handler = checkpoint_handler.load_checkpoint(checkpoint_path)
```