import collections

import numpy as np
import torch
from trw.transforms import transforms
from trw.transforms import affine


def rand_n_2(n_min_max):
    """
    Create random values for a list of min/max pairs

    Args:
        n_min_max: a matrix of size N * 2 representing N points where `n_min_max[:, 0]` are the minimum values
            and `n_min_max[:, 1]` are the maximum values

    Returns:
        `N` random values in the defined interval

    Examples:
        To return 3 random values in interval [-1..10], [-2, 20], [-3, 30] respectively:
        >>> n_min_max = np.asarray([[-1, 10], [-2, 20], [-3, 30]])
        >>> rand_n_2(n_min_max)
    """
    assert len(n_min_max.shape) == 2
    assert n_min_max.shape[1] == 2, 'must be min/max'

    r = np.random.rand(len(n_min_max))
    scaled_r = np.multiply(r, (n_min_max[:, 1] - n_min_max[:, 0])) + n_min_max[:, 0]
    return scaled_r


class TransformAffine(transforms.TransformBatchWithCriteria):
    """
    Transform an image using a random affine (2D or 3D) transformation.

    Only 2D or 3D supported transformation.
    """
    def __init__(self, translation_min_max, scaling_min_max, rotation_radian_min_max, criteria_fn=None):
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_3_or_above

        self.criteria_fn = criteria_fn
        self.rotation_radian_min_max = rotation_radian_min_max
        self.scaling_min_max = scaling_min_max
        self.translation_min_max = translation_min_max

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

        # normalize the transformation. We want a Dim * 2 matrix
        translation = np.asarray(self.translation_min_max)
        if len(translation.shape) == 0:  # single value
            translation = np.asarray([[-translation, translation]] * data_dim)
        elif len(translation.shape) == 1:  # min/max
            assert len(translation) == 2
            translation = np.repeat([[translation[0], translation[1]]], data_dim, axis=0)
        else:
            assert translation.shape == (data_dim, 2), 'expected [(min_x, max_x), (min_y, max_y), ...]'

        tfms = []
        for n in range(data_shape[0]):
            random_translation_offset = rand_n_2(translation)
            matrix_translation = affine.affine_transformation_translation(random_translation_offset)
            matrix_transform = matrix_translation
            matrix_transform = affine.to_voxel_space_transform(matrix_transform, data_shape[1:])
            tfms.append(matrix_transform)

        tfms = torch.stack(tfms, dim=0)

        new_batch = collections.OrderedDict()
        for name, value in batch.items():
            if name in features_names:
                new_batch[name] = affine.affine_transform(value, tfms)
            else:
                new_batch[name] = value
        return new_batch
