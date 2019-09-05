import torchvision
from .utils import named_dataset


@named_dataset(names=['images', 'targets'])
class MNIST(torchvision.datasets.MNIST):
    """
    .. note: the data root can be set using the environment variable `TRW_DATA_ROOT`
    """
    pass


@named_dataset(names=['images', 'targets'])
class FakeData(torchvision.datasets.FakeData):
    pass
