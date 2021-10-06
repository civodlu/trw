from typing import Sequence, Callable, List, Union, Optional
from typing_extensions import Literal

from .deform import deform_image_random

from ..transforms import transforms
from ..basic_typing import Batch


class TransformRandomDeformation(transforms.TransformBatchWithCriteria):
    """
    Transform an image using a random deformation field.

    Only 2D or 3D supported transformation.

    The gradient can be back-propagated through this transform.
    """
    def __init__(
            self,
            control_points: Union[int, Sequence[int]],
            max_displacement: Optional[Union[float, Sequence[float]]] = None,
            criteria_fn: Callable[[Batch], List[str]] = None,
            interpolation: Literal['linear', 'nearest'] = 'linear',
            padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
            align_corners: bool = False):
        """

        Args:
            control_points: the control points spread on the image at regularly
                spaced intervals with random `max_displacement` magnitude
            max_displacement: specify the maximum displacement of a control point. Range [-1..1]. If None, use
                the moving volume shape and number of control points to calculate appropriate small deformation
                field
            interpolation: the interpolation of the image with displacement field
            padding_mode: how to handle data outside the volume geometry
            align_corners: should be False. The (0, 0) is the center of a voxel
            criteria_fn: a function to select applicable features in a batch
        """

        self.interpolation = interpolation
        self.align_corners = align_corners
        self.max_displacement = max_displacement
        self.control_points = control_points
        self.padding_mode = padding_mode
        self.criteria_fn = criteria_fn

        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=self._transform
        )

    def _transform(self, features_names, batch):
        data_shape = batch[features_names[0]].shape
        data_dim = len(data_shape) - 2  # remove `N` and `C` components
        assert data_dim == 2 or data_dim == 3, f'only 2D or 3D data handled. Got={data_dim}'

        for name in features_names[1:]:
            # make sure the data is correct: we must have the same dimensions (except `C`)
            # for all the images
            feature = batch[name]
            feature_shape = feature.shape[2:]
            assert feature_shape == data_shape[2:], f'joint features transformed must have the same dimension. ' \
                                                    f'Got={feature_shape}, expected={data_shape[2:]}'
            assert feature.shape[0] == data_shape[0]

        images = [batch[name] for name in features_names]
        deformed_images = deform_image_random(
            images,
            control_points=self.control_points,
            max_displacement=self.max_displacement,
            interpolation=self.interpolation,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )

        # copy features that are not images
        new_batch = {name: value for name, value in zip(features_names, deformed_images)}
        for name, value in batch.items():
            if name not in new_batch:
                new_batch[name] = value
        return new_batch
