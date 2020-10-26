import functools
from unittest import TestCase
import trw
import numpy as np
import torch
from trw.transforms import resample_3d, SpatialInfo, random_fixed_geometry_within_geometries


class TestTransformResample(TestCase):
    def test_resample_numpy_copy_subvolume(self):
        """
        Simply copy a sub-volume
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (2, 3, 4), (4, 5, 6), (1, 1, 1))
        t_expected = t[2:4, 3:5, 4:6]
        assert (t_r == t_expected).all()

    def test_resample_numpy_with_background(self):
        """
        Simply copy a sub-volume with background voxels
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (2, 3, 4), (4, 5, 20), (1, 1, 1), constant_value=42)
        t_r_valid = t_r[:, :, :2]
        t_expected = t[2:4, 3:5, 4:6]
        assert (t_r_valid == t_expected).all()

        # outside voxel should be background value
        t_r_background = t_r[:, :, 4:]
        assert (t_r_background == 42).all()

    def test_resample_numpy_id(self):
        """
        identity
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (0, 0, 0), (10, 9, 8), (1, 1, 1), constant_value=42, order=0)
        assert (t == t_r).all()

    def test_resample_numpy_interpolate_x(self):
        """
        Interpolate in x axis
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (0, 0, 0), (10, 9, 8), (1, 1, 0.49999), constant_value=42, order=0)
        assert t_r.shape == (10, 9, 8 * 2)
        for z in range(t_r.shape[0]):
            for y in range(t_r.shape[1]):
                for x in range(t_r.shape[2] - 1):  # avoid the background value at the end
                    v = t_r[z, y, x]
                    v_expected = t[z, y, x // 2]
                    assert v == v_expected

    def test_resample_torch_id(self):
        """
        identity
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8))
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (0, 0, 0), (10, 9, 8), (1, 1, 1), constant_value=42, order=0)
        assert (t == t_r).all()

    @staticmethod
    def get_spatial_info_generic(batch, name, geometry) -> SpatialInfo:
        return geometry

    def test_transform_background_constant_numpy(self):
        batch = {
            'v1': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)).numpy(),
            'v2': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)).numpy(),
            'other': 'other_value'
        }

        get_spatial_info = functools.partial(TestTransformResample.get_spatial_info_generic,
                                             geometry=SpatialInfo(origin=(0, 0, 0), spacing=(1, 1, 1), shape=(10, 9, 8)))

        transform = trw.transforms.TransformResample(
            get_spatial_info_from_batch_name=get_spatial_info,
            resampling_geometry=SpatialInfo(origin=(0, 0, 0), spacing=(1, 1, 1), shape=(1, 9, 18)),
            constant_background_value=42
        )

        batch_transformed = transform(batch)
        assert len(batch_transformed) == 3
        assert batch_transformed['other'] == 'other_value'
        assert batch_transformed['v1'].shape == (1, 1, 1, 9, 18)
        assert batch_transformed['v2'].shape == (1, 1, 1, 9, 18)

        # voxel within FoV of the volumes
        assert (batch_transformed['v1'][0, 0, :, :, :8] == batch['v1'][0, 0, 0:1, :, :8]).all()
        assert (batch_transformed['v2'][0, 0, :, :, :8] == batch['v2'][0, 0, 0:1, :, :8]).all()

        # voxels in the background
        assert (batch_transformed['v1'][0, 0, 0:1, :, 8:] == 42).all()
        assert (batch_transformed['v2'][0, 0, 0:1, :, 8:] == 42).all()

    def test_transform_background_volume_dependent(self):
        batch = {
            'v1': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'v2': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'other': 'other_value'
        }

        get_spatial_info = functools.partial(TestTransformResample.get_spatial_info_generic,
                                             geometry=SpatialInfo(origin=(0, 0, 0), spacing=(1, 1, 1), shape=(10, 9, 8)))

        transform = trw.transforms.TransformResample(
            get_spatial_info_from_batch_name=get_spatial_info,
            resampling_geometry=SpatialInfo(origin=(0, 0, 0), spacing=(1, 1, 1), shape=(1, 9, 18)),
            constant_background_value={'v1': 42, 'v2': 43}
        )

        batch_transformed = transform(batch)
        assert len(batch_transformed) == 3
        assert batch_transformed['other'] == 'other_value'
        assert batch_transformed['v1'].shape == (1, 1, 1, 9, 18)
        assert batch_transformed['v2'].shape == (1, 1, 1, 9, 18)

        # voxel within FoV of the volumes
        assert (batch_transformed['v1'][0, 0, :, :, :8] == batch['v1'][0, 0, 0:1, :, :8]).all()
        assert (batch_transformed['v2'][0, 0, :, :, :8] == batch['v2'][0, 0, 0:1, :, :8]).all()

        # voxels in the background
        assert (batch_transformed['v1'][0, 0, 0:1, :, 8:] == 42).all()
        assert (batch_transformed['v2'][0, 0, 0:1, :, 8:] == 43).all()

    @staticmethod
    def get_random_geometry(dict_of_g):
        g = SpatialInfo(
            shape=np.random.randint(0, 8, size=[3]),
            origin=np.random.uniform(-4, 6, size=[3]),
            spacing=np.random.uniform(0.5, 1.5, size=[3]),
        )
        return g

    def test_transform_resampling_random_dependent(self):
        """
        Use a functors to randomly generate a resampling geometry.

        All volumes MUST use the same geometry.
        """
        batch = {
            'v1': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'v2': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'other': 'other_value'
        }

        get_spatial_info = functools.partial(TestTransformResample.get_spatial_info_generic,
                                             geometry=SpatialInfo(origin=(0, 0, 0), spacing=(1, 1, 1), shape=(10, 9, 8)))

        transform = trw.transforms.TransformResample(
            get_spatial_info_from_batch_name=get_spatial_info,
            resampling_geometry=TestTransformResample.get_random_geometry,
        )

        batch_transformed = transform(batch)
        assert len(batch_transformed) == 3
        assert batch_transformed['other'] == 'other_value'
        assert batch_transformed['v1'].shape == batch_transformed['v2'].shape

    def test_transform_resampling_random_fixed(self):
        batch = {
            'v1': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'v2': torch.arange(1 * 1 * 1).reshape((1, 1, 1, 1, 1)),
        }

        get_spatial_info = functools.partial(TestTransformResample.get_spatial_info_generic,
                                             geometry=SpatialInfo(origin=(0, 0, 0),
                                                                  spacing=(1, 1, 1),
                                                                  shape=(10, 9, 8)))

        transform = trw.transforms.TransformResample(
            get_spatial_info_from_batch_name=get_spatial_info,
            resampling_geometry=functools.partial(random_fixed_geometry_within_geometries, fixed_geometry_shape=(3, 3, 3), fixed_geometry_spacing=(1, 1, 1)),
        )

        batch_transformed = transform(batch)
        assert len(batch_transformed) == 2
        assert batch_transformed['v1'].shape == batch_transformed['v2'].shape
