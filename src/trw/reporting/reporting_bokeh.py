import math
import functools
import collections
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, TableColumn, DataTable, HTMLTemplateFormatter, Select, \
    CategoricalColorMapper
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show, curdoc
from bokeh.plotting import figure
import os
import numpy as np
from bokeh.transform import transform
from tornado import web
from trw.reporting.table_sqlite import get_tables_name_and_role, get_table_data, \
    get_data_types_and_clean_data
from trw.reporting import utilities
import sqlite3
from bokeh.server.server import Server

#
# table roles:
#   - data_samples: extract of data with possibly model output. Data exploration + model output correlations
#


class Object:
    pass


def create_default_reporting_options(embedded=True):
    o = Object()
    o.image_size = 64
    o.font_size = 19
    o.data_samples = Object()
    o.data_samples.display_tabular = True
    o.data_samples.display_scatter = True
    o.db_root = None
    o.embedded = embedded
    o.style = Object()
    o.style.color_by_line_width = 2
    o.style.scatter_aspect_ratio = 1.5
    o.style.tool_window_size_x = 200
    o.style.tool_window_size_y = 500
    o.style.sorted_legend = True
    o.style.category_margin = 0.2
    return o


def process_data_samples__tabular(options, data, data_types):
    """
    Create a tabular panel of the data & types

    Args:
        options:
        data: a dictionary of (key, values)
        data_types: a dictionary of (key, type) indicating special type of ``key``

    Returns:
        a panel
    """
    image_style = """
                img
                { 
                    height:%dpx; 
                    width:%dpx;
                } 
                """ % (options.image_size, options.image_size)

    template = f"""
                <div>
                <style>
                {image_style}
                </style>
                <img
                    src=<%= value %>
                ></img>
                </div>
                """

    with_images = False
    columns = []
    for key in data.keys():
        type = data_types.get(key)
        if type is not None and 'BLOB_IMAGE' in type:
            with_images = True
            c = TableColumn(field=key, title=key, formatter=HTMLTemplateFormatter(template=template))
        else:
            c = TableColumn(field=key, title=key)
        columns.append(c)

    data_source = ColumnDataSource(data)

    row_height = options.font_size
    if with_images:
        row_height = options.image_size
    data_table = DataTable(source=data_source, columns=columns, row_height=row_height)
    return Panel(child=column(data_table, sizing_mode='stretch_both'), title='Tabular')


def scatter(all_data, groups, scatter_name, is_discrete_unordered):
    if is_discrete_unordered:
        # discrete types, without ordering. Split the data into sub-groups
        # need to use global `unique_values` so that the groups are synchronized
        scatter_values = all_data[scatter_name]
        unique_values = set(scatter_values)
        unique_values = sorted(unique_values)

        final_groups = []
        for group in groups:
            data = all_data[scatter_name][group]

            current_groups = []
            for id, value in enumerate(unique_values):
                indices = np.where(data == value)
                current_groups.append(group[indices])
            final_groups.append(current_groups)
        return final_groups, [{'value': value, 'type': 'unordered'} for value in unique_values]
    else:
        raise NotImplementedError()


def group_coordinate(options, group, group_layout_y, group_layout_x):
    """
    Create a coordinate system for each group, independently of the other groups
    """
    if group_layout_y['type'] == 'unordered' and group_layout_x['type'] == 'unordered':
        nb_columns = math.ceil(math.sqrt(len(group)))
        coords_x = (np.arange(len(group)) % nb_columns) * options.image_size
        coords_y = (np.arange(len(group)) // nb_columns) * options.image_size
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    raise NotImplementedError()


def sync_groups_coordinates(options, data_x, data_y, groups, groups_layout_y, groups_layout_x):
    """
    Synchronize the different group coordinate systems
    """
    for x in range(groups.shape[1] - 1):
        groups_at_x = np.concatenate(groups[:, x])
        min_value_for_next_group = data_x[groups_at_x].max() + options.image_size + \
                                   options.style.category_margin * options.image_size

        for y in range(groups.shape[0]):
            group = groups[y, x + 1]
            if len(group) == 0:
                # there are no sample for this group, simply skip it
                continue
            min_group = data_x[group].min()
            shift = max(min_value_for_next_group - min_group, 0)
            data_x[group] += shift

    for y in range(groups.shape[0] - 1):
        groups_at_y = np.concatenate(groups[y, :])
        min_value_for_next_group = data_y[groups_at_y].max() + options.image_size + \
                                   options.style.category_margin * options.image_size

        for x in range(groups.shape[1]):
            group = groups[y + 1, x]
            if len(group) == 0:
                # there are no sample for this group, simply skip it
                continue
            min_group = data_y[group].min()
            shift = max(min_value_for_next_group - min_group, 0)
            data_y[group] += shift


def render_data(options, fig, data, data_types, scatter_x, scatter_y, color_by, color_scheme, icon, label_with, layout, data_source):
    print('render command STARTED')
    nb_data = utilities.len_batch(data)
    fig.title.text = f'Samples selected: {nb_data}'
    fig.renderers.clear()
    fig.yaxis.visible = False
    fig.xaxis.visible = False

    if 'data_x' not in data_source.column_names:
        data_source.add(np.zeros(nb_data, np.float32), 'data_x')
        data_source.add(np.zeros(nb_data, np.float32), 'data_y')
        data_source.add([None] * nb_data, 'color_by')

    fig.xaxis.axis_label = scatter_x.value
    fig.yaxis.axis_label = scatter_y.value

    if len(scatter_x.value) > 0 and scatter_x.value != 'None':
        scatter_x = scatter_x.value
        assert scatter_x in data, f'column not present in data! c={scatter_x}'
    else:
        scatter_x = None

    if len(scatter_y.value) > 0 and scatter_y.value != 'None':
        scatter_y = scatter_y.value
        assert scatter_y in data, f'column not present in data! c={scatter_y}'
    else:
        scatter_y = None

    if scatter_x is not None:
        groups, groups_layout_x = scatter(data, [np.arange(nb_data)], scatter_x, is_discrete_unordered=True)
        groups = groups[0]
    else:
        # consider all the points as x
        groups = [np.arange(nb_data)]
        groups_layout_x = [{'value': '', 'type': 'unordered'}]

    if scatter_y is not None:
        groups, groups_layout_y = scatter(data, groups, scatter_y, is_discrete_unordered=True)
    else:
        # consider all the points as y
        groups = [[g] for g in groups]
        groups_layout_y = [{'value': '', 'type': 'unordered'}]

    if len(color_by.value) > 0 and color_by.value != 'None':
        color_by = color_by.value
    else:
        color_by = None

    # in case we have array of 3D instead of 2D+list (if samples have exactly the same in the 2 groups)
    # we need to transpose only the first and second but NOT the third
    groups = np.asarray(groups)
    groups_transpose = list(range(len(groups.shape)))
    tmp = groups_transpose[0]
    groups_transpose[0] = groups_transpose[1]
    groups_transpose[1] = tmp
    groups = groups.transpose(groups_transpose)

    data_x = np.zeros(nb_data, dtype=np.float)
    data_y = np.zeros(nb_data, dtype=np.float)
    for y in range(groups.shape[0]):
        for x in range(groups.shape[1]):
            group = groups[y, x]
            group_layout_y = groups_layout_y[y]
            group_layout_x = groups_layout_x[x]

            c = group_coordinate(options, group, group_layout_y, group_layout_x)
            data_x[group] = c[:, 0]
            data_y[group] = c[:, 1]

    sync_groups_coordinates(options, data_x, data_y, groups, groups_layout_y, groups_layout_x)

    data_source.data['data_x'] = data_x
    data_source.data['data_y'] = data_y

    if len(icon.value) > 0 and icon.value != 'Icon':
        # the images do NOT support tooltips yet in bokeh, instead, overlay a transparent rectangle
        # that will display the tooltip
        fig.image_url(url=icon.value, x='data_x', y='data_y', h=options.image_size, w=options.image_size, anchor="center", source=data_source)
        fig.rect(x='data_x', y='data_y', width=options.image_size, height=options.image_size, width_units='data', height_units='data', source=data_source, fill_alpha=0, line_alpha=0)
    else:
        fig.oval(x='data_x', y='data_y', width=options.image_size, height=options.image_size, width_units='data', height_units='data', source=data_source)

    # remove the "color by" legend panel
    while len(layout.children) > 2:
        layout.children.pop()

    if color_by is not None:
        colors_all_values = data[color_by]
        color_unique_values = list(set(colors_all_values))
        if options.style.sorted_legend:
            color_unique_values = sorted(color_unique_values)
        color_unique_palette_ints = np.random.randint(0, 255, [len(color_unique_values), 3])
        color_unique_palette_hexs = ['#%0.2X%0.2X%0.2X' % tuple(c) for c in color_unique_palette_ints]

        color_mapper = CategoricalColorMapper(
            factors=color_unique_values,
            palette=color_unique_palette_hexs)

        data_source.data['color_by'] = colors_all_values

        fig.rect(
            source=data_source,
            x='data_x',
            y='data_y',
            width=options.image_size - options.style.color_by_line_width,  # TODO data space mixed with screen space
            height=options.image_size - options.style.color_by_line_width,
            line_color=transform('color_by', color_mapper),
            fill_color=None,
            line_width=options.style.color_by_line_width,
         )

        # could not use the simple legend mechanism (was messing up the figure.
        # Instead resorting to another figure). This has the added benefit that we can
        # zoom in and out of the legend in case we have many categories
        sub_fig = figure(
            title='Labels',
            width=options.style.tool_window_size_x, height=options.style.tool_window_size_y,
            x_range=(-0.1, 1), y_range=(0, 26),
            tools='pan,reset',
            toolbar_location='above')
        y_text = 25 - np.arange(0, len(color_unique_values))
        sub_fig.circle(x=0, y=y_text, size=15, color=color_unique_palette_hexs)
        sub_fig.axis.visible = False
        sub_fig.xgrid.visible = False
        sub_fig.ygrid.visible = False
        sub_fig.text(x=0, y=y_text, text=color_unique_values,  x_offset=15, y_offset=6 // 2, text_font_size='6pt')
        layout.children.append(sub_fig)
    else:
        data_source.data['color_by'] = [None] * nb_data

    if scatter_x and groups_layout_x[0]['type'] == 'unordered':
        ticks = []
        ticks_label = []
        for x in range(groups.shape[1]):
            coords_x = data_x[np.concatenate(groups[:, x])]
            if len(coords_x) > 0:
                min_value = coords_x.min()
                max_value = coords_x.max()
                center_value = int(min_value + max_value) // 2

                ticks += [center_value]
                ticks_label += [groups_layout_x[x]['value']]

        fig.xaxis.visible = True
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = {t: name for t, name in zip(ticks, ticks_label)}
        fig.xaxis.major_label_orientation = math.pi / 4
    else:
        fig.xaxis.visible = False

    if scatter_y and groups_layout_y[0]['type'] == 'unordered':
        ticks = []
        ticks_label = []
        for y in range(groups.shape[0]):
            coords_y = data_y[np.concatenate(groups[y, :])]
            if len(coords_y) > 0:
                min_value = coords_y.min()
                max_value = coords_y.max()
                center_value = int(min_value + max_value) // 2

                ticks += [center_value]
                ticks_label += [groups_layout_y[y]['value']]

        fig.yaxis.visible = True
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = {t: name for t, name in zip(ticks, ticks_label)}
        fig.yaxis.major_label_orientation = math.pi / 4
    else:
        fig.yaxis.visible = False

    print('render command DONE')


def process_data_samples__scatter(options, data, data_types):
    print('process_data_samples__scatter')
    values = ['None'] + list(sorted(data.keys()))
    scatter_x_axis = Select(title='Scatter X Axis', options=values, value='targets')  # TODO REMOVE DEFAULT
    scatter_y_axis = Select(title='Scatter Y Axis', options=values, value='split_name')  # TODO REMOVE DEFAULT
    color_by = Select(title='Color by', options=values)
    color_scheme = Select(title='Color scheme', options=['Discrete', 'Grey'])
    binning_x_axis = Select(title='Binning X Axis', options=values)
    binning_y_axis = Select(title='Binning Y Axis', options=values)
    label_with = Select(title='Label with', options=values)

    icon_values = [v for v, t in data_types.items() if 'BLOB_IMAGE' in t] + ['Icon']
    icon = Select(title='Display with', options=icon_values, value=icon_values[0])

    controls_list = [
        scatter_x_axis,
        scatter_y_axis,
        color_by,
        color_scheme,
        binning_x_axis,
        binning_y_axis,
        icon,
        label_with
    ]
    controls = column(controls_list, width=options.style.tool_window_size_x, height=options.style.tool_window_size_y)
    controls.sizing_mode = "fixed"


    tooltips = [
        ("data_x", "$data_x"),
        ("data_x", "$data_y"),
    ]
    tooltips.append(('index', '$index'))
    #tooltips.append(('image', '$image'))  # TODO must format the numpy text array!!
    #for key in data.keys():
    #    tooltips.append((key, f'${key}'))

    tools = 'pan,wheel_zoom,reset'
    f = figure(
        title='',
        tools=tools,
        active_scroll='wheel_zoom',
        match_aspect=True,
        aspect_ratio=options.style.scatter_aspect_ratio,
        aspect_scale=1,
        toolbar_location='above',
        tooltips=tooltips,
    )
    f.xaxis.axis_label = scatter_x_axis.value
    f.yaxis.axis_label = scatter_y_axis.value
    f.xgrid.visible = False
    f.ygrid.visible = False

    # make sure the hover tools are always on the rendered glyphs!!!
    # If not, the hover tool will work initially, but as soon as the
    # `f.renderers.clear()` is called, the hover tool will never work
    # again
    f.hover.renderers = f.renderers

    layout = row(controls, f, sizing_mode='scale_height')

    data_source = ColumnDataSource(data)

    update = functools.partial(
        render_data,
        options=options,
        fig=f,
        data=data,
        data_types=data_types,
        scatter_x=scatter_x_axis,
        scatter_y=scatter_y_axis,
        color_by=color_by,
        color_scheme=color_scheme,
        icon=icon,
        label_with=label_with,
        layout=layout,
        data_source=data_source,
    )

    for control in controls_list:
       control.on_change('value', lambda attr, old, new: update())

    update()

    return Panel(child=layout, title='Scatter')


def normalize_data(options, data):
    """
    Normalize the data

    The following operations are performed:
    - convert to numpy arrays
    - removed type column and return a specific `type` dictionary
    - normalize the path according to deployment: if static html, nothing to do
        if deployed, we MUST add the `app name` (the top folder name containing the SQL DB)
        as root
    """
    d = collections.OrderedDict()
    for name, values in data.items():
        d[name] = np.asarray(values)
    types = get_data_types_and_clean_data(d)

    if options.embedded:
        appname = os.path.basename(os.path.dirname(options.db_root))
        for n, t in types.items():
            if 'BLOB_' in t:
                # it is a blob, meaning that it was saved on the local HD
                d[n] = np.core.defchararray.add(np.asarray([f'{appname}/']), d[n])
    return d, types


def process_data_samples(options, connexion, name):
    """

    Args:
        options:
        connexion:
        name:

    Returns:

    """
    data, types = normalize_data(options, get_table_data(connexion, name))

    tabs = []
    if options.data_samples.display_tabular:
        tabs.append(process_data_samples__tabular(options, data, types))

    if options.data_samples.display_scatter and options.embedded:
        # here we require some python logic, so we need to have a bokeh
        # server running to display this view
        tabs.append(process_data_samples__scatter(options, data, types))

    return Tabs(tabs=tabs)


def report(sql_database_path, options, doc=None):
    """

    Args:
        sql_database_path:

    Returns:

    """
    options.db_root = sql_database_path
    connexion = sqlite3.connect(sql_database_path)
    name_roles = get_tables_name_and_role(connexion)
    root = os.path.dirname(sql_database_path)

    if not options.embedded:
        output_file(os.path.join(root, 'index.html'), title=os.path.basename(root))

    if doc is None:
        doc = curdoc()
    doc.title = os.path.basename(root)

    panels = []
    for name, role in name_roles:
        if role == 'data_samples':
            data_table = process_data_samples(options, connexion, name)
            panel = Panel(child=data_table, title=name)
            panels.append(panel)

    tabs = Tabs(tabs=panels)
    if options.embedded:
        doc.add_root(tabs)

    else:
        show(tabs)

    return doc


def run_server(path_to_db):
    def fn(doc):
        return report(sql_database_path=path_to_db, options=create_default_reporting_options(embedded=True), doc=doc)

    app_1 = Application(FunctionHandler(fn))

    # here we expect to have a folder with the following structure
    # root/{app_name}/static/{...} for all the static resources (e.g., images, numpy arrays)
    app_directory = os.path.dirname(path_to_db)
    app_name = os.path.basename(app_directory)
    apps = {f'/{app_name}': app_1}
    server = Server(apps, num_procs=1)

    # set up the `static` mapping
    handlers = [
        (
            f'/{app_name}/static/(.*)',
            web.StaticFileHandler,
            {'path': os.path.join(app_directory, 'static')},
        )
    ]
    server._tornado.add_handlers(r".*", handlers)

    # start the server
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == '__main__':
    # create an embedded (with responsive python in the backend)
    run_server('/path/reporting_sqlite.db')

    # run a static HTML page (i.e., the more complicated views requiring python v=callbacks will be disabled)
    #report('/path/reporting_sqlite.db', options=create_default_reporting_options(embedded=False))
