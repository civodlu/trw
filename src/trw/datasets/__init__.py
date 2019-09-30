from .mnist import create_mnist_datasset
from .cifar10 import create_cifar10_dataset
from .chunked_dataset import chunk_samples, DatasetChunked, read_pickle_simple_one, write_pickle_simple, read_whole_chunk, create_chunk_sequence, create_chunk_reservoir, chunk_name
from .chunked_dataset import _read_whole_chunk_sequence  # TODO REMOVE

from .dataset_fake_symbols import create_fake_symbols_datasset, _random_location, _random_color, _add_shape, _create_image, _noisy
from .dataset_fake_symbols_2d import create_fake_symbols_2d_datasset
from .dataset_fake_symbols_3d import create_fake_symbols_3d_datasset

