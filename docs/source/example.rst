Example: classification of the MNIST dataset
============================================

MNIST dataset
-------------

We will be using the MNIST classification task as an example to give an overview of the framework. The purpose 
of this task is to classify a 28x28 white and black image into one of the ten possible digits. We have access to
55,000 training images to train the model parameters, 5,000 images for validation to select the best hyper-parameters
this task and 10,000 images to assess the model.

.. image:: images/mnistdigits.png
    :align: center
    :alt: MNIST examples
    :height: 300px


Specify and train the model
---------------------------

In this section we define a classification model, but first, let's import commonly required modules:

.. testcode::

	import trw
	import torch.nn as nn
	import torch.nn.functional as F
	import numpy as np

Using the native PyTorch API, we define our model. To specify that a node
should be used as a classification unit, we use the :class:`trw.train.OutputClassification`.
By default it will use the multi-class cross entropy loss:

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

Evaluation of the model
-----------------------

Inspecting the input data
^^^^^^^^^^^^^^^^^^^^^^^^^

Basic Statistics
^^^^^^^^^^^^^^^^

Example errors
^^^^^^^^^^^^^^
