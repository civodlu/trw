Torch Research Workflows
========================

.. image:: https://dev.azure.com/civodlu/trw/_apis/build/status/civodlu.trw?branchName=master
   :target: https://dev.azure.com/civodlu/trw/_build/results
   
.. image:: https://readthedocs.org/projects/trw/badge/?version=latest
   :target: https://trw.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Purpose
-------

The aim of this library is to simplify the process of building, optimizing, testing and debugging
deep learning models using PyTorch as well as providing implementations of some of the latest
research papers. Extensibility is kept in mind so that it is easy to customize the framework for
your particular needs.

Some key features of the framework:

* Easy to use, flexible and extensible API to build simple & complex models 
* Model debugging (e.g., activation statistics of each layer, gradient norm for each layer, embedding visualization)
* Model understanding and result analysis (e.g., attention maps, confusion matrix, ROC curves, model comparisons, errors)
* Support hyper-parameter optimization (random search, hyperband) and analysis
* Architecture learning (DARTS & evolutionary algorithms)
* Keep track of the results for retrospective analysis and model selection

Installation / Usage
--------------------

To install use pip:

    $ pip install trw


Or clone the repo:

    $ git clone https://github.com/civodlu/trw.git
    
    $ python setup.py install
    
Example on the MNIST dataset
----------------------------

Let's import the required modules:

.. testcode::

	import trw
	import torch.nn as nn
	import torch.nn.functional as F
	import numpy as np


Then we can define a neural network:

.. testcode::

	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.conv1 = nn.Conv2d(1, 20, 5, 2)
			self.fc1 = nn.Linear(20 * 6 * 6, 500)
			self.fc2 = nn.Linear(500, 10)
			self.relu_fc1 = nn.ReLU()
			self.relu_conv1 = nn.ReLU()

		def forward(self, batch):
			# a batch should be a dictionary of features
			x = batch['images'] / 255.0

			x = self.relu_conv1(self.conv1(x))
			x = F.max_pool2d(x, 2, 2)
			x = x.view(x.shape[0], -1)
			x = self.relu_fc1(self.fc1(x))
			x = self.fc2(x)

			# here we create a softmax output that will use
			# the `targets` feature as classification target
			return {
				'softmax': trw.train.OutputClassification(x, 'targets')
			}


Finally, we can create a :class:`trw.train.Trainer` to start the training and evaluation:

.. testcode::

	# configure and run the training/evaluation
	options = trw.train.create_default_options(num_epochs=10)
	trainer = trw.train.Trainer()

	model, results = trainer.fit(
		options,
		inputs_fn=lambda: trw.datasets.create_mnist_datasset(),
		run_prefix='mnist_cnn',
		model_fn=lambda options: Net(),
		optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
			datasets=datasets, model=model, learning_rate=0.1))

	# calculate statistics of the final epoch
	output = results['outputs']['mnist']['test']['softmax']
	accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
	assert accuracy >= 0.95

