import functools
import collections
from bokeh.models.widgets import Tabs
from trw.reporting.bokeh_ui import BokehUi
from trw.reporting.normalize_data import normalize_data
from trw.reporting.table_sqlite import get_table_data, get_table_number_of_rows, get_tables_name_and_role, table_create, \
    get_metadata_name, table_insert
from trw.reporting import safe_lookup, len_batch


def create_aliased_table(connection, name, aliased_table_name):
    metadata_name = get_metadata_name(name)
    alias_role = f'alias##{aliased_table_name}'

    cursor = connection.cursor()
    table_create(cursor, metadata_name, ['table_role TEXT'], primary_key=None)
    table_insert(cursor, metadata_name, names=['table_role'], values=[alias_role])

    # create an empty table
    table_create(cursor, name, ['dummy_column TEXT'], primary_key=None)


def get_data_normalize_and_alias(options, connection, name):
    """

    Retrieve data from SQL, normalize the data and resolve aliasing (

    Returns:
        tuple (data, types, type_categories, alias)
    """
    table_name_roles = get_tables_name_and_role(connection.cursor())
    table_name_roles = dict(table_name_roles)
    role = table_name_roles.get(name)
    assert role, f'table={name} doesn\'t have a role!'

    if 'alias##' in role:
        splits = role.split('##')
        assert len(splits) == 2, 'alias is not well formed. Expeced ``alias##aliasname``'
        alias = splits[1]
        # in an aliased table, use the ``name`` to pickup the correct config from the ``options``
        data, types, type_categories = normalize_data(options, get_table_data(connection, alias), table_name=name)
    else:
        alias = None
        data, types, type_categories = normalize_data(options, get_table_data(connection, name), table_name=name)

    return data, types, type_categories, alias


class TabsDynamicData(BokehUi):
    """
    Helper class to manage updates of the underlying SQL data for a given table
    """
    def __init__(self, doc, options, connection, name, role, creator_fn):
        data, types, type_categories, alias = get_data_normalize_and_alias(options, connection, name)
        tabs = creator_fn(options, name, role, data, types, type_categories)
        tabs_ui = []
        for tab in tabs:
            assert isinstance(tab, BokehUi), 'must be a ``BokehUi`` based!'
            tabs_ui.append(tab.get_ui())

        if len(tabs_ui) > 1:
            ui = Tabs(tabs=tabs_ui)
        else:
            assert len(tabs_ui) > 0
            ui = tabs_ui[0]
        super().__init__(ui=ui)

        self.last_update_data_size = len_batch(data)
        doc.add_periodic_callback(functools.partial(self.update,
                                                    options=options,
                                                    connection=connection,
                                                    name=name,
                                                    tabs=tabs),
                                  options.data.refresh_time * 1000)

    def update(self, options, connection, name, tabs):
        number_of_rows = get_table_number_of_rows(connection, name)
        if number_of_rows != self.last_update_data_size:  # different number of rows, data was changed!
            self.last_update_data_size = number_of_rows
            # discard `0`, the table creation is not part of a
            # transaction, the table is being populated
            if number_of_rows > 0:
                data, types, type_categories = normalize_data(options, get_table_data(connection, name), table_name=name)
                keep_last_n_rows = safe_lookup(options.config, name, 'data', 'keep_last_n_rows')
                if keep_last_n_rows is not None:
                    data_trimmed = collections.OrderedDict()
                    for name, values in data.items():
                        data_trimmed[name] = values[-keep_last_n_rows:]
                    data = data_trimmed

                for tab in tabs:
                    tab.update_data(options, name, data, types, type_categories)
