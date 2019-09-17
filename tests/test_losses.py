import trw
import numpy as np
import torch
from unittest import TestCase


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
