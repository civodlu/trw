Example: classification of the MNIST dataset
********************************************

MNIST dataset
=============

We will be using the MNIST classification task as an example to give an overview of the framework. The purpose 
of this task is to classify a 28x28 white and black image into one of the ten possible digits. We have access to
55,000 training images to train the model parameters, 5,000 images for validation to select the best hyper-parameters
this task and 10,000 images to assess the model.

.. image:: images/mnistdigits.png
    :align: center
    :alt: MNIST examples
    :height: 300px


Specify and train the model
===========================

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

			# Here we create a softmax output that will use
			# the `targets` feature as classification target
			return {
				'softmax': trw.train.OutputClassification(x, 'targets')
			}


Finally, we can create a :class:`trw.train.Trainer` to start the training and evaluation:

.. testcode::

	# Configure and run the training/evaluation
	options = trw.train.create_default_options(num_epochs=10)
	trainer = trw.train.Trainer()

	model, results = trainer.fit(
		options,
		inputs_fn=lambda: trw.datasets.create_mnist_datasset(),
		run_prefix='mnist_cnn',
		model_fn=lambda options: Net(),
		optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
			datasets=datasets, model=model, learning_rate=0.1))

	# Calculate statistics of the final epoch
	output = results['outputs']['mnist']['test']['softmax']
	accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
	assert accuracy >= 0.95

Evaluation of the model
=======================

By default `TRW` keeps track of useful information to check input inputa data, evaluate and debug
the model. By default, the following folders will be created:

.. code-block::

	mnist_cnn_r0
		├── random_samples
		├── augmentations
		├── tensorboard
		├── history
		├── lr_recorder
		├── history
		├── worst_samples_by_epoch
		├── errors
		| best_history.txt
		| last.model
		| last.model.result
		| softmax-mnist-test-cm.png
		| softmax-mnist-test-report.txt
		| softmax-mnist-train-cm.png
		| softmax-mnist-train-report.txt
		| trainer.txt
		

Inspecting the input data
-------------------------

The :class:`trw.train.Trainer` will output random samples in `mnist_cnn_r0/random_samples/mnist`. 
Features of a sample that can be natively exported in a meaningful format (e.g., .png for an image). 
For example, `mnist_test_s0_e0.png` will be a random image from the test split:

.. figure:: images/mnist_test_s0_e0_images.png
    :align: center


Other metadata that cannot be exported to a known format will be summarized in a text file. For example, 
`mnist_test_s0_e0.txt` will store metadata such as sample ID, target class:

.. code-block::
	
	targets=5
	sample_uid=6573
	dataset_name=mnist
	split_name=test
	term_softmax_output=6
	targets_str=5 - five


Inspecting the data augmentations
---------------------------------

To make sure the data augmentation is behaving as expected, it is useful to visualize them. By default 
augmentations will be stored in the `mnist_cnn_r0/augmentations/mnist` folder. Internally, 
:class:`trw.train.SequenceArray` will create a unique ID per sample that will be used to keep track
of the augmentations.

Basic Statistics
----------------

At the end of the training, meaningful statistics will be gathered:

* ROC and AUC for binary classification,
* Confusion matrix,
* Accuracy, precision, recall, F1 score, most common errors,
* Evolution of accuracy and losses during the training.

.. figure:: images/softmax-mnist-test-cm.png
    :align: center


Example errors
--------------

Using the callback :class:`trw.train.CallbackWorstSamplesByEpoch`, a selected number of
samples with errors will be exported. Another useful view is to display the errors by epoch 
using :class:`trw.train.CallbackWorstSamplesByEpoch` and inspect the samples that
are the most frequently classified and in particular in the training split.  These are the 
errors the classifier has the most difficulty assimilating and often reveal the outliers. Here 
is an example below on the train split:

.. figure:: images/mnist-train-softmax-e40.png
    :align: center
	
    The samples are displayed on the x-axis (one per pixel) and y-axis shows the epochs. `Red` 
    indicates a sample with high loss while yellow indicates samples with low loss. Samples are sorted
    by overall loss.
	
	
Here are the most difficult examples to classify. This can be used quickly identify outliers:

.. figure:: images/outliers.png
    :align: center
	
    Examples of outliers and annotation mistakes spotted using :class:`trw.train.CallbackWorstSamplesByEpoch`


Embedding analysis
------------------

:class:`trw.train.CallbackTensorboardEmbedding` allows to export an intermediate tensor (or commonly referred to
as `embedding`) to the tensorboard embedding tab. This can be useful to understand what the model considers as similar
samples and possibly detect common trends.

.. figure:: images/mnist_embedding.png
    :align: center


Explainable decisions
---------------------
TBD

Hyper-parameter selection & visualization
-----------------------------------------
TBD

Archtecture search
------------------
TBD

Model Export
------------

Finally, the model is stored as PyTorch model and exported to a `onnx` format. This allows interoperability
between major deep learning frameworks (e.g., for production).