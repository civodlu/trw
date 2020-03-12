import trw
import numpy as np
import torch
from unittest import TestCase


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TestLosses(TestCase):
    def test_losses_dice_single_binary_sample(self):
        found_channel0 = [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]

        found_channel1 = [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        targets = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ]

        t1 = torch.from_numpy(np.asarray([found_channel0, found_channel1], dtype=np.float32).reshape((1, 2, 5, 6)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 5, 6)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None)
        l = loss(t1, t2)

        # calculate the mask intersection
        # we must have a mask for background & foreground
        num_intersection_0 = 4
        num_intersection_1 = 6

        # calculate the number of pixels ON of segmentation of background & foreground
        num_output_0 = 18
        num_output_1 = 12

        num_target_0 = 10
        num_target_1 = 20

        # calculate the expected dice score
        expected_dice_0 = 2 * num_intersection_0 / (num_output_0 + num_target_0)
        expected_dice_1 = 2 * num_intersection_1 / (num_output_1 + num_target_1)

        # transform this to a loss that we want to minimize
        expected_loss = 1.0 - (expected_dice_0 + expected_dice_1) / 2.0

        assert abs(expected_loss - float(l)) < 1e-4

    def test_losses_dice_multiple_binary_sample(self):
        targets = [
            [[1, 1, 0],
             [1, 1, 0]],

            [[0, 1, 1],
             [0, 1, 1]],
        ]

        sample_0 = [
            # channel 0
            [[1, 1, 1],
             [0, 0, 0]],

            # channel 1
            [[0, 0, 0],
             [1, 1, 1]],
        ]

        sample_1 = [
            # channel 0
            [[1, 1, 0],
             [1, 1, 0]],

            # channel 1
            [[0, 0, 1],
             [0, 0, 1]],
        ]

        t1 = torch.from_numpy(np.asarray([sample_0, sample_1], dtype=np.float32).reshape((2, 2, 2, 3)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((2, 2, 3)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None)
        l = loss(t1, t2)
        assert len(l.shape) == 1
        assert l.shape[0] == 2

        sample_0_num_intersection_0 = 1
        sample_0_num_intersection_1 = 2

        sample_1_num_intersection_0 = 2
        sample_1_num_intersection_1 = 2

        sample_0_num_output_0 = 3
        sample_0_num_output_1 = 3

        sample_1_num_output_0 = 4
        sample_1_num_output_1 = 2

        sample_0_num_target_0 = 2
        sample_0_num_target_1 = 4

        sample_1_num_target_0 = 2
        sample_1_num_target_1 = 4

        sample_0_expected_dice_0 = 2 * sample_0_num_intersection_0 / (sample_0_num_output_0 + sample_0_num_target_0)
        sample_0_expected_dice_1 = 2 * sample_0_num_intersection_1 / (sample_0_num_output_1 + sample_0_num_target_1)

        sample_1_expected_dice_0 = 2 * sample_1_num_intersection_0 / (sample_1_num_output_0 + sample_1_num_target_0)
        sample_1_expected_dice_1 = 2 * sample_1_num_intersection_1 / (sample_1_num_output_1 + sample_1_num_target_1)

        # transform this to a loss that we want to minimize
        samples_0_expected_loss = 1.0 - (sample_0_expected_dice_0 + sample_0_expected_dice_1) / 2.0
        samples_1_expected_loss = 1.0 - (sample_1_expected_dice_0 + sample_1_expected_dice_1) / 2.0

        assert abs(samples_0_expected_loss - float(l[0])) < 1e-4
        assert abs(samples_1_expected_loss - float(l[1])) < 1e-4

    def test_losses_dice_single_multichannel_sample(self):
        found_channel0 = [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
        ]

        found_channel1 = [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        found_channel2 = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ]

        targets = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1],
        ]

        t1 = torch.from_numpy(np.asarray([found_channel0, found_channel1, found_channel2], dtype=np.float32).reshape((1, 3, 5, 6)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 5, 6)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None)
        l = loss(t1, t2)

        # calculate the mask intersection
        # we must have a mask for background & foreground
        num_intersection_0 = 3
        num_intersection_1 = 6
        num_intersection_2 = 2

        # calculate the number of pixels ON of segmentation of background & foreground
        num_output_0 = 16
        num_output_1 = 12
        num_output_2 = 2

        num_target_0 = 9
        num_target_1 = 18
        num_target_2 = 3

        # calculate the expected dice score
        expected_dice_0 = 2 * num_intersection_0 / (num_output_0 + num_target_0)
        expected_dice_1 = 2 * num_intersection_1 / (num_output_1 + num_target_1)
        expected_dice_2 = 2 * num_intersection_2 / (num_output_2 + num_target_2)

        # transform this to a loss that we want to minimize
        expected_loss = 1.0 - (expected_dice_0 + expected_dice_1 + expected_dice_2) / 3.0

        assert abs(expected_loss - float(l)) < 1e-4

    def test_dice_by_class_perfect(self):
        found_channel0 = [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        found_channel1 = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]

        found_channel2 = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]

        targets = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
        ]

        t1 = torch.from_numpy(np.asarray([found_channel0, found_channel1, found_channel2], dtype=np.float32).reshape((1, 3, 4, 3)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 4, 3)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True)
        l = loss(t1, t2)
        assert abs(float(l.sum()) - 3) < 1e-3

    def test_dice_by_class_2samples(self):
        e1_found_channel0 = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
        ]

        e1_found_channel1 = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 1],
        ]

        e2_found_channel0 = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]

        e2_found_channel1 = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]

        targets = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]

        t1 = torch.from_numpy(np.asarray([e1_found_channel0, e1_found_channel1, e2_found_channel0, e2_found_channel1], dtype=np.float32).reshape((2, 2, 4, 3)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 4, 3)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True)
        l = loss(t1, t2).numpy()

        expected_dice_class0 = [2 * 9 / (9 + 10), 2 * 9 / (9 + 9)]
        expected_dice_class1 = [2 * 2 / (2 + 3), 2 * 3 / (3 + 3)]
        assert abs(l[0] - (sum(expected_dice_class0) / 2)) < 1e-5
        assert abs(l[1] - (sum(expected_dice_class1) / 2)) < 1e-5

    def test_focal_loss_binary_id(self):
        # in this configuration, the cross entropy loss and the focal loss MUST be identical
        targets = torch.from_numpy(np.asarray([0, 1, 1, 0, 1], dtype=np.int64))
        outputs = torch.from_numpy(np.asarray([[0.5, 0.1], [0.1, 0.9], [0.1, 0.8], [0.6, 0.1], [0.3, 0.6]], dtype=np.float32))
        alpha = np.asarray([1.0, 1.0], dtype=np.float32)

        loss_focal_fn = trw.train.LossFocalMulticlass(alpha=alpha, gamma=0.0)
        loss_focal = loss_focal_fn(outputs, targets)

        loss_ce_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ce = loss_ce_fn(outputs, targets)

        # gamma == 0 means we MUST retrun cross entropy
        assert float((loss_focal - loss_ce).abs().max()) <= 1e-6

    def test_focal_loss_binary_ordering(self):
        # make sure the more confident classes get less loss weigth
        targets = torch.from_numpy(np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        outputs = torch.from_numpy(np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32))
        alpha = torch.from_numpy(np.asarray([1.0, 1.0], dtype=np.float32))

        loss_focal_fn = trw.train.LossFocalMulticlass(alpha=alpha, gamma=10.0)
        loss_focal = trw.train.to_value(loss_focal_fn(outputs, targets))
        assert len(loss_focal) == len(targets)

        ce_loss = trw.train.to_value(torch.nn.CrossEntropyLoss(reduction='none')(outputs, targets))

        # rescale the focal loss to have a similar max range
        loss_focal_scaled = max(ce_loss) / max(loss_focal) * loss_focal

        assert loss_focal_scaled[0] * 10 < ce_loss[0]
        assert loss_focal_scaled[5] * 10 < ce_loss[5]

        assert loss_focal_scaled[1] * 3 < ce_loss[1]
        assert loss_focal_scaled[4] * 3 < ce_loss[4]

    def test_focal_loss_multiclass_2d(self):
        targets = torch.ones([10, 5, 5], dtype=torch.int64)
        outputs = torch.ones([10, 3, 5, 5], dtype=torch.float32)

        loss_focal_fn = trw.train.LossFocalMulticlass(gamma=1.0)
        loss_focal = trw.train.to_value(loss_focal_fn(outputs, targets))
        assert len(loss_focal) == len(targets)
