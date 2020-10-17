from typing import List

from trw.transforms import transforms
import collections

from trw.typing import Batch


class TransformCompose(transforms.Transform):
    """
    Sequentially apply a list of transformations
    """
    def __init__(self, transforms: List[transforms.Transform]):
        """

        Args:
            transforms: a list of :class:`trw.transforms.Transform`
        """
        assert isinstance(transforms, collections.Sequence), '`transforms` must be a sequence!'
        self.transforms = transforms

    def __call__(self, batch: Batch) -> Batch:
        for transform in self.transforms:
            batch = transform(batch)
        return batch
