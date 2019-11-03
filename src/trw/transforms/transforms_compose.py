from trw.transforms import transforms
import collections


class TransformCompose(transforms.Transform):
    """
    Sequentially apply a list of transformations
    """
    def __init__(self, transforms):
        """

        Args:
            transforms (list): a list of :class:`trw.transforms.Transform`
        """
        assert isinstance(transforms, collections.Sequence), '`transforms` must be a sequence!'
        self.transforms = transforms

    def __call__(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch
