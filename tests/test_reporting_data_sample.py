import os
import sqlite3
import tempfile
from unittest import TestCase
import trw.reporting
import numpy as np
from bokeh.models import Div, DataTable
from bokeh.plotting import Figure

from trw.reporting import TableStream, export_sample
from trw.reporting.reporting_bokeh import normalize_data, DataCategory, process_data_samples
from trw.reporting.table_sqlite import get_table_data


def make_table(cursor, table_name, table_role, batch):
    tmp_folder = tempfile.mkdtemp()
    output_dir = os.path.join(tmp_folder, 'static')
    os.makedirs(os.path.join(output_dir, table_name))

    table_stream = TableStream(cursor, table_name, table_role)
    export_sample(tmp_folder, table_stream, 'basename', batch)
    return tmp_folder


class TestReporting(TestCase):
    def test_data_normalization_column_expansion(self):
        # multi dimensional numpy arrays can be expanded as a single columns
        # to facilitate analysis
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'numpy_arrays_int_5': np.random.randint(0, 10, [3, 5]),
        }

        table_name = 'table_name'
        tmp_folder = make_table(cursor, table_name, 'table_role', batch)

        data = get_table_data(cursor, table_name)
        options = trw.reporting.create_default_reporting_options()
        options.db_root = os.path.join(tmp_folder, 'test.db')

        normalized_data, types, type_categories = normalize_data(options, data, table_name)
        assert len(normalized_data) == 5
        assert 'numpy_arrays_int_5_0' in normalized_data
        assert 'numpy_arrays_int_5_1' in normalized_data
        assert 'numpy_arrays_int_5_2' in normalized_data
        assert 'numpy_arrays_int_5_3' in normalized_data
        assert 'numpy_arrays_int_5_4' in normalized_data
        assert type_categories['numpy_arrays_int_5_4'] == DataCategory.DiscreteOrdered

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
            'column_to_be_removed': ['p1', 'p2', 'p3'],
        }

        table_name = 'table_name'
        tmp_folder = make_table(cursor, table_name, 'table_role', batch)

        subsampling_factor = 2 / 3
        data = get_table_data(cursor, table_name)
        options = trw.reporting.create_default_reporting_options()
        options.config = {
            'table_name': {
                'data': {
                    'subsampling_factor': subsampling_factor,
                    'remove_columns': ['column_to_be_removed']
                }
            }
        }
        options.data.unpack_numpy_arrays_with_less_than_x_columns = 3
        options.db_root = os.path.join(tmp_folder, 'test.db')

        # must have sub-sampled the data
        normalized_data, types, type_categories = normalize_data(options, data, table_name)
        assert len(normalized_data['constant']) == 2, 'subsampling failed!'

        # must have exactly the batch keys
        assert len(batch) == len(normalized_data) + 1
        for key in normalized_data.keys():
            assert key in batch
        assert 'column_to_be_removed' not in normalized_data.keys()

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

    def test_table_data_samples(self):
        # make sure we can render the samples tabs with tabular and scatter data
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'constant': 0,
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
            'split': 'train',
            'column_to_be_removed': ['p1', 'p2', 'p3'],
        }

        table_name = 'table_name'
        tmp_folder = make_table(cursor, table_name, 'table_role', batch)

        options = trw.reporting.create_default_reporting_options()
        options.db_root = os.path.join(tmp_folder, 'test.db')

        tabs = process_data_samples(options, connection, 'table_name')

        assert len(tabs.tabs) == 2  # must have the `tabular` and `scatter` tabs

        #
        # Check the tabular tab
        #
        assert len(tabs.tabs[0].child.children) == 2  # must have a DataTable, Div
        table = tabs.tabs[0].child.children[0]
        assert isinstance(table, DataTable)
        div = tabs.tabs[0].child.children[1]
        assert isinstance(tabs.tabs[0].child.children[1], Div)  # DIV to configure extra CSS
                                                                # for the table (text rotation)
        assert div.visible is False

        assert len(table.source.column_names) == len(batch)

        #
        # Check the scatter tab
        #
        head = tabs.tabs[1].child.children
        assert len(head[0].children) >= 7  # all the tools

        figures = head[1].children
        assert len(figures) == 1  # should have a single figure
        assert isinstance(figures[0], Figure)
        assert len(figures[0].renderers) > 0  # if not, failure!

    def test_table_data_samples__default_config__discrete_discrete(self):
        # make sure we can configure defaults
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'constant': 0,
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
            'split': 'train',
        }

        table_name = 'table_name'
        tmp_folder = make_table(cursor, table_name, 'table_role', batch)

        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_integers',
                    'Scatter Y Axis': 'strings',
                    'Binning X Axis': 'value_integers',
                    'Color by': 'value_continuous',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        tabs = process_data_samples(options, connection, 'table_name')

        #
        # Check the scatter tab
        #
        head = tabs.tabs[1].child.children
        assert len(head) == 4  # Tools, Scatter 1+2+3

        assert head[0].children[0].value == 'value_integers'
        assert head[0].children[1].value == 'strings'
        assert head[0].children[2].value == 'value_continuous'
        assert head[0].children[3].value == 'Viridis256'
        assert head[0].children[4].value == 'value_integers'
        assert head[0].children[5].value == 'Dot'
        print('DONE')

    def test_table_data_samples__scatter_and_discrete_configs(self):
        # make sure we can configure defaults
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'constant': 0,
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
            'split': 'train',
        }

        table_name = 'table_name'
        tmp_folder = make_table(cursor, table_name, 'table_role', batch)

        #
        # X: continuous, Y: discrete
        #
        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_continuous',
                    'Scatter Y Axis': 'value_integers',
                    'Color by': 'strings',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        tabs = process_data_samples(options, connection, 'table_name')
        head = tabs.tabs[1].child.children
        assert len(head) == 3  # Tools, Fig, Figure-color-legend
        assert len(head[1].children[0].renderers) > 0

        #
        # Y: continuous, X: discrete
        #
        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_integers',
                    'Scatter Y Axis': 'value_continuous',
                    'Color by': 'strings',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        tabs = process_data_samples(options, connection, 'table_name')
        head = tabs.tabs[1].child.children
        assert len(head) == 3  # Tools, Fig, Figure-color-legend
        assert len(head[1].children[0].renderers) > 0

        #
        # Y: continuous, X: continuous
        #
        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_continuous',
                    'Scatter Y Axis': 'value_continuous',
                    'Color by': 'strings',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        tabs = process_data_samples(options, connection, 'table_name')
        head = tabs.tabs[1].child.children
        assert len(head) == 3  # Tools, Fig, Figure-color-legend
        assert len(head[1].children[0].renderers) > 0
