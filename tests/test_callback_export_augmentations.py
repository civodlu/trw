from unittest import TestCase
import trw.train
import numpy as np
import os


def transorm_add_random(batch):
    batch_len = trw.train.len_batch(batch)
    r = np.random.random_integers(0, 5, batch_len)
    batch['values'] = batch['values'] + r
    return batch


class TestCallbackExportAugmentations(TestCase):
    def test_simple_augmentations(self):
        data = {
            'values': np.arange(1000)
        }

        datasets = {'dataset1': {'train': trw.train.SequenceArray(split=data).map(transorm_add_random)}}

        nb_samples = 20
        callback = trw.train.CallbackExportAugmentations(nb_samples=nb_samples, keep_samples=True)
        options = trw.train.create_default_options()
        options['workflow_options']['current_logging_directory'] = os.path.join(options['workflow_options']['current_logging_directory'], 'unit_test')
        callback(options, None, None, None, None, datasets, None, None)

        self.assertTrue(len(callback.list_of_samples_by_uid) == 1)
        uids_by_sample = callback.list_of_samples_by_uid[0]
        self.assertTrue(len(uids_by_sample) == nb_samples)

        for sample_uid, augmentations in uids_by_sample.items():
            for augmentation in augmentations:
                value = augmentation['values']
                self.assertTrue(abs(sample_uid - value) <= 5)


