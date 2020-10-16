import collections

import functools
from numbers import Number
from typing import Sequence, Callable, Union, Any, Dict

import trw.utils
from trw.transforms import transforms
from trw.transforms import crop
import numpy as np


def _transform_resize_modulo_crop_pad(
        features_names,
        batch,
        multiple_of,
        mode,
        padding_mode,
        padding_constant_value):

    resized_arrays = []
    arrays = [batch[name] for name in features_names]
    for a in arrays[1:]:
        assert a.shape == arrays[0].shape, f'selected tensors MUST have the same size. ' \
                                           f'Expected={a.shape}, got={arrays[0].shape}'

    if len(features_names) > 0:
        shape = arrays[0].shape[2:]

        if isinstance(multiple_of, Number):
            multiple_of = [multiple_of] * (len(shape))
        multiple_of = np.asarray(multiple_of)

        if mode == 'pad':
            raise NotImplementedError('TODO IMPLEMENT')
        elif mode == 'crop':
            shape_extra = arrays[0].shape[2] % multiple_of
            resized_arrays = crop.transform_batch_random_crop_joint(arrays, [None] + list(shape - shape_extra))
        else:
            raise NotImplementedError(f'mode={mode} is not handled!')

    new_batch = collections.OrderedDict(zip(features_names, resized_arrays))
    for feature_name, feature_value in batch.items():
        if feature_name not in features_names:
            # not in the transformed features, so copy the original value
            new_batch[feature_name] = feature_value

    return new_batch


class TransformResizeModuloCropPad(transforms.TransformBatchWithCriteria):
    """
    Resize tensors by padding or cropping the tensor so its shape is a multiple of a size.

    This can be particularly helpful in encoder-decoder architecture with skip connection which can impose
    constraints on the input shape (e.g., the input must be a multiple of 32 pixels).

    Args:
        multiple_of: a sequence of size `len(array.shape)-2` such that shape % multiple_of == 0. To achieve this,
            the tensors will be padded or cropped.
        criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
            the original feature is returned
        padding_mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        mode: one of `crop`, `pad`. If `pad`, the selected tensors will be padded to achieve the
            size tensor.shape % multiple_of == 0. If `crop`, the selected tensors will be cropped
            instead with a randomly selected cropping position.

    Returns:
        dictionary with the selected tensors cropped or padded to the appropriate size
    """
    def __init__(
            self,
            multiple_of: Union[int, Sequence[int]],
            criteria_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
            #mode: Literal['crop', 'pad'] = 'crop',  # TODO python 3.8
            mode: str = 'crop',
            #padding_mode: Literal['edge', 'constant', 'symmetric'] = 'edge',  # TODO python 3.8
            padding_mode: str = 'edge',
            padding_constant_value: int = 0):

        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_3_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(
                _transform_resize_modulo_crop_pad,
                padding_mode=padding_mode,
                mode=mode,
                padding_constant_value=padding_constant_value,
                multiple_of=multiple_of)
         )
        self.criteria_fn = transforms.criteria_is_array_3_or_above

