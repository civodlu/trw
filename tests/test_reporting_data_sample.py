import os
import sqlite3
import tempfile
from unittest import TestCase
import trw.reporting
import numpy as np

from trw.reporting import TableStream, export_sample
from trw.reporting.reporting_bokeh import normalize_data, DataCategory
from trw.reporting.table_sqlite import get_table_data


def make_table(cursor, table_name, table_role, batch):
    tmp_folder = tempfile.mkdtemp()
    output_dir = os.path.join(tmp_folder, 'static')
    os.makedirs(os.path.join(output_dir, table_name))

    table_stream = TableStream(cursor, table_name, table_role)
    export_sample(tmp_folder, table_stream, 'basename', batch)
    return tmp_folder


class TestReporting(TestCase):
    def test_data_normalization(self):
        # publish data the same way the application would:
        # export samples to a database, then read the data
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'constant': 0,
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_10': np.random.randint(0, 10, [3, 10]),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
            'split': 'train',
        }

        table_name = 'table_name'
        tmp_folder = make_table(cursor, table_name, 'table_role', batch)

        subsampling_factor = 2 / 3
        data = get_table_data(cursor, table_name)
        options = trw.reporting.create_default_reporting_options()
        options.config = {
            'table_name': {
                'data': {
                    'subsampling_factor': subsampling_factor
                }
            }
        }
        options.data.unpack_numpy_arrays_with_less_than_x_columns = 3
        options.db_root = os.path.join(tmp_folder, 'test.db')

        # must have sub-sampled the data
        normalized_data, types, type_categories = normalize_data(options, data, table_name)
        assert len(normalized_data['constant']) == 2, 'subsampling failed!'

        # must have exactly the batch keys
        assert len(batch) == len(normalized_data)
        for key in batch.keys():
            assert key in normalized_data

        # for BLOB_IMAGE, must have appended the `appname` (folder of the datbase name)
        appname = os.path.basename(os.path.dirname(options.db_root))
        assert appname in normalized_data['images'][0]

        # images must be served from a static directory
        assert 'static' in normalized_data['images'][0]

        assert type_categories['images'] == DataCategory.Other
        assert type_categories['numpy_arrays'] == DataCategory.Other  # too many dimensions
        assert type_categories['numpy_arrays_int_10'] == DataCategory.Other
        assert type_categories['numpy_arrays_int_1'] == DataCategory.DiscreteOrdered
        assert type_categories['value_continuous'] == DataCategory.Continuous
        assert type_categories['value_integers'] == DataCategory.DiscreteOrdered
        assert type_categories['strings'] == DataCategory.DiscreteUnordered
        assert type_categories['split'] == DataCategory.DiscreteUnordered

        assert types['numpy_arrays'] == 'BLOB_NUMPY'
        assert types['images'] == 'BLOB_IMAGE_PNG'
        assert types['numpy_arrays_int_10'] == 'BLOB_NUMPY'
