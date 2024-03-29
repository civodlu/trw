from bokeh.models import TabPanel as Panel
from bokeh.models.layouts import Tabs
import os
from .bokeh_ui import BokehUi
from .table_sqlite import get_tables_name_and_role
import json

from ..utils import recursive_dict_update


class TabsDynamicHeader(BokehUi):
    """
    Helper class to manage updates of the underlying SQL tables & JSON configuration.

    This class will monitor at defined time interval if a given SQL table or JSON file was modified. If
    so, a callback will be executed (e.g., to update the UI).
    """
    def __init__(self, doc, options, connection, creator_fn):
        self.tabs = Tabs(max_height=options.frame_size_y, max_width=options.frame_size_x)
        self.existing_tables = []
        self.options = options
        self.connection = connection
        self.creator_fn = creator_fn
        self.config_timestamp = None

        super().__init__(ui=self.tabs)
        doc.add_periodic_callback(self.update, options.data.refresh_time * 1000)

        # initial update
        self.update()

    def update(self):
        # first, check the config was not modified
        config_location = self.options.db_root.replace('.db', '.json')
        if os.path.exists(config_location):
            time_stamp = os.path.getmtime(config_location)
            if time_stamp != self.config_timestamp:
                self.config_timestamp = time_stamp
                with open(config_location, 'r') as f:
                    f_str = f.read()
                new_config = json.loads(f_str)

                # make sure we keep the original config and only update
                # changed or new elements
                recursive_dict_update(self.options.config, new_config)

        all_names = []
        new_name_roles = []
        name_roles = get_tables_name_and_role(self.connection)
        for name, role in name_roles:
            all_names.append(name)
            if name not in self.existing_tables:
                new_name_roles.append((name, role))

        if len(new_name_roles) > 0:
            self.update_new_tables(new_name_roles, self.creator_fn)

        self.existing_tables = all_names

    def update_new_tables(self, name_roles, creator_fn):
        for name, role in name_roles:
            panel = creator_fn(name, role)
            assert isinstance(panel, Panel)
            self.tabs.tabs.append(panel)
