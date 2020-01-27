import unittest
import trw
import os


class TestCallbackExportBestHistory(unittest.TestCase):
    def test_basic(self):
        def make_step(dataset_name, split_name, task_name, measurements_dict):
            return {
                dataset_name: {
                    split_name: {
                        task_name: measurements_dict
                    }
                }
            }

        history = [
            make_step('dataset_1', 'split_1', 'task_1', {'measurement_1': 0.1, 'measurement_2': 0.2, 'measurement_3': 0.2}),
            make_step('dataset_1', 'split_1', 'task_1', {'measurement_1': 0.2, 'measurement_2': 0.15, 'measurement_3': 0.2}),
            make_step('dataset_1', 'split_1', 'task_1', {'measurement_1': 0.3, 'measurement_2': 0.5, 'measurement_3': 0.05, 'measurement_4': 1.0}),
        ]

        options = trw.train.create_default_options()
        callback = trw.train.CallbackExportBestHistory(filename='test.txt')

        callback(options, history, None, None, None, None, None, None)

        expected_file = os.path.join(options['workflow_options']['logging_directory'], 'test.txt')
        with open(expected_file, 'r') as f:
            lines = f.readlines()

        results = {}
        for line in lines:
            [name_value, step] = line.strip().split(',')
            step_value = step.split('=')[1]
            kvp = name_value.split('=')
            assert len(kvp) == 2
            results[kvp[0]] = (kvp[1], step_value)

        assert len(results) == 4
        assert results['dataset_1_split_1_task_1_measurement_1'] == ('0.1', '0')
        assert results['dataset_1_split_1_task_1_measurement_2'] == ('0.15', '1')
        assert results['dataset_1_split_1_task_1_measurement_3'] == ('0.05', '2')
        assert results['dataset_1_split_1_task_1_measurement_4'] == ('1.0', '2')


if __name__ == '__main__':
    unittest.main()
