import warnings
from enum import Enum

import math
import functools
import collections
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, TableColumn, DataTable, HTMLTemplateFormatter, Select, \
    CategoricalColorMapper, HoverTool, LinearColorMapper, ColorBar, FixedTicker, NumberFormatter
from bokeh.models.widgets import Panel, Tabs, Div
from bokeh.io import output_file, show, curdoc
from bokeh.plotting import figure, Figure
import os
import numpy as np
from bokeh.transform import transform
from tornado import web
from trw.reporting.table_sqlite import get_tables_name_and_role, get_table_data, \
    get_data_types_and_clean_data
from trw.reporting import utilities, safe_lookup, len_batch
import sqlite3
from bokeh.server.server import Server
import json
#
# table roles:
#   - data_samples: extract of data with possibly model output. Data exploration + model output correlations
#


class Object:
    pass


def create_default_reporting_options(embedded=True, config={}):
    o = Object()
    o.image_size = 64
    o.font_size = 19
    o.data_samples = Object()
    o.data_samples.display_tabular = True
    o.data_samples.display_scatter = True
    o.data_samples.max_numpy_display = 10  # if array below this size, the content will be displayed
    o.db_root = None
    o.embedded = embedded
    o.style = Object()
    o.style.color_by_line_width = 1
    o.style.scatter_aspect_ratio = 1.5
    o.style.tool_window_size_x = 200
    o.style.tool_window_size_y = 500
    o.style.sorted_legend = True
    o.style.category_margin = 0.2
    o.style.scatter_continuous_factor = 10

    o.data = Object()
    o.data.unpack_numpy_arrays_with_less_than_x_columns = 15

    o.config = config

    # config must follow this scheme
    # {
    #   `table_name`: {
    #       `option1`: `value1`
    #   }
    # }

    # the different options will depend on the table role.
    #
    # - for ALL tables:
    #    {
    #       'data' : {
    #           'remove_columns': ['column_name1']
    #           'subsampling_factor': 1.0
    #       }
    #
    # - For role `data_samples`:
    #    {
    #       'default':
    #           'Scatter X Axis': value,
    #           'Scatter Y Axis': value,
    #           'Color by': value,
    #           'Color scheme': value,
    #           'Binning X Axis': value,
    #           'Binning Y Axis': value,
    #           'Label with': value,
    #           'Display with': value,
    #    }
    #
    #

    return o


def process_data_samples__tabular(options, data, data_types, type_categories):
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
            type_category = type_categories[key]
            table_kargs = {}
            if type_category == DataCategory.Continuous:
                table_kargs['formatter'] = NumberFormatter(format='0,0[.000]')

            c = TableColumn(field=key, title=key, **table_kargs)
        columns.append(c)

    # filter out data
    filtered_data = filter_large_data(options, data)
    data_source = ColumnDataSource(filtered_data)

    # custom CSS to slightly rotate the column header
    # and draw text outside their respective columns
    # to improve readability. TODO That could be improved
    div = Div(text="""
    <style>
    .trw_reporting_table .slick-header-column {
            background-color: transparent;
            background-image: none !important;
            transform: 
                rotate(-10deg)
          }
          
    .bk-root .slick-header-column.ui-state-default {
        height: 40px;
        overflow: visible;
        vertical-align: bottom;
        line-height: 4.4;
    }
    </style>
    """)
    div.visible = False  # hide the div to avoid position issues

    row_height = options.font_size
    if with_images:
        row_height = options.image_size
    data_table = DataTable(
        source=data_source,
        columns=columns,
        row_height=row_height,
        css_classes=["trw_reporting_table"])
    return Panel(child=column(data_table, div, sizing_mode='stretch_both'), title='Tabular')


def scatter(all_data, groups, scatter_name, type_category):
    if type_category in (DataCategory.DiscreteUnordered, DataCategory.DiscreteOrdered):
        # discrete types, without ordering. Split the data into sub-groups
        # need to use global `unique_values` so that the groups are synchronized
        scatter_values = all_data[scatter_name]
        unique_values = set(scatter_values)
        if type_category == DataCategory.DiscreteOrdered:
            unique_values = sorted(unique_values)

        final_groups = []
        for group in groups:
            data = np.asarray(all_data[scatter_name])[group]

            current_groups = []
            for id, value in enumerate(unique_values):
                indices = np.where(data == value)
                current_groups.append(group[indices])
            final_groups.append(current_groups)
        return final_groups, [{'value': value, 'type': 'discrete'} for value in unique_values]
    elif type_category == DataCategory.Continuous:
        all_scatter_values = np.asarray(all_data[scatter_name])[np.concatenate(groups)]
        value_min = all_scatter_values.min()
        value_max = all_scatter_values.max()
        return [[g] for g in groups], [{'value_min': value_min, 'value_max': value_max, 'type': 'continuous', 'scatter_name': scatter_name}]
    else:
        raise NotImplementedError()


def group_coordinate(options, data, group, group_layout_y, group_layout_x):
    """
    Create a coordinate system for each group, independently of the other groups
    """
    if group_layout_y['type'] == 'discrete' and group_layout_x['type'] == 'discrete':
        nb_columns = math.ceil(math.sqrt(len(group)))
        coords_x = (np.arange(len(group)) % nb_columns) * options.image_size
        coords_y = (np.arange(len(group)) // nb_columns) * options.image_size
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    if group_layout_y['type'] == 'continuous' and group_layout_x['type'] == 'discrete':
        nb_rows = len(group)
        scatter_name = group_layout_y['scatter_name']
        coords_x = np.random.rand(nb_rows) * options.image_size * 0.75
        coords_y = np.asarray(data[scatter_name])[group]
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    if group_layout_x['type'] == 'continuous' and group_layout_y['type'] == 'discrete':
        nb_rows = len(group)
        scatter_name = group_layout_x['scatter_name']
        coords_y = np.random.rand(nb_rows) * options.image_size * 0.75
        coords_x = np.asarray(data[scatter_name])[group]
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    if group_layout_x['type'] == 'continuous' and group_layout_y['type'] == 'continuous':
        scatter_name_x = group_layout_x['scatter_name']
        scatter_name_y = group_layout_y['scatter_name']
        coords_x = np.asarray(data[scatter_name_x])[group]
        coords_y = np.asarray(data[scatter_name_y])[group]
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    raise NotImplementedError()


def sync_groups_coordinates(options, data_x, data_y, groups, groups_layout_y, groups_layout_x):
    """
    Up to that point, each group had sample coordinates calculated independently from
    each other. Here consolidate all the subcoordinate system into a global one
    """
    if groups_layout_x[0]['type'] == 'discrete':
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
    elif groups_layout_x[0]['type'] == 'continuous':
        # for continuous axis, we expect a single group in Y and coordinates spread between [0, pixel_y]
        assert groups.shape[1] == 1, 'only handle shape 1xN'
        pixel_x = options.style.scatter_continuous_factor * options.image_size
        factor = pixel_x / (groups_layout_x[0]['value_max'] - groups_layout_x[0]['value_min']) + 1e-6
        indices = np.concatenate(groups[:, 0])
        data_x[indices] = (data_x[indices] - groups_layout_x[0]['value_min']) * factor

        groups_layout_x[0]['coord_max'] = pixel_x
        groups_layout_x[0]['coord_min'] = 0
    else:
        raise NotImplementedError()

    if groups_layout_y[0]['type'] == 'discrete':
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
    elif groups_layout_y[0]['type'] == 'continuous':
        # for continuous axis, we expect a single group in Y and coordinates spread between [0, pixel_y]
        assert groups.shape[0] == 1, 'only handle shape 1xN'
        pixel_y = options.style.scatter_continuous_factor * options.image_size
        factor = pixel_y / (groups_layout_y[0]['value_max'] - groups_layout_y[0]['value_min']) + 1e-6
        indices = np.concatenate(groups[0])
        data_y[indices] = (data_y[indices] - groups_layout_y[0]['value_min']) * factor

        groups_layout_y[0]['coord_max'] = pixel_y
        groups_layout_y[0]['coord_min'] = 0
    else:
        raise NotImplementedError()


def filter_large_data(options, data):
    """
    remove very large data, this will be communicated to the client and this is
    very slow!
    """
    filtered_data = collections.OrderedDict()
    for name, values in data.items():
        if isinstance(values[0], np.ndarray) and values[0].size > options.data_samples.max_numpy_display:
            # data is too large and take too long to display so remove it
            filtered_data[name] = ['...'] * len(values)
            warnings.warn(f'numpy array ({name}) is too large ({values[0].size}) and has been discarded')
            continue
        filtered_data[name] = values
    return filtered_data


def render_data(
        options, data, data_types, type_categories, scatter_x, scatter_y, binning_x_axis,
        color_by, color_scheme, icon, label_with, layout):
    previous_figure = None

    while len(layout.children) >= 2:
        layout.children.pop()

    if len(binning_x_axis.value) > 0 and binning_x_axis.value != 'None':
        scatter_values = data[binning_x_axis.value]
        unique_values = set(list(scatter_values))
        unique_values = list(sorted(unique_values))

        groups = []
        for value in unique_values:
            group = np.where(np.asarray(scatter_values) == value)
            groups.append(group)
    else:
        nb_data = utilities.len_batch(data)
        groups = [[np.arange(nb_data)]]

    # remove very large data, this will be communicated to the client and this is
    # very slow!
    data = filter_large_data(options, data)

    x_range = None
    y_range = None
    for group_n, group in enumerate(groups):
        # copy the data: there were some display issues if not
        subdata = {name: list(np.asarray(value)[group[0]]) for name, value in data.items()}
        data_source = ColumnDataSource(subdata)
        nb_data = len(group[0])
        group = [np.arange(nb_data)]

        layout_children = render_data_frame(
            f'fig{group_n}', options, data_source,
            subdata,
            group, data_types, type_categories, scatter_x,
            scatter_y, color_by, color_scheme, icon, label_with, previous_figure)

        if x_range is None:
            x_range = layout_children[0].children[0].x_range
            y_range = layout_children[0].children[0].y_range
        else:
            layout_children[0].children[0].x_range = x_range
            layout_children[0].children[0].y_range = y_range

        for c in layout_children:
            layout.children.append(c)


def render_data_frame(fig_name, options, data_source, data, groups, data_types, type_categories, scatter_x, scatter_y, color_by, color_scheme, icon, label_with, previous_figure):
    print('render command STARTED, nb_samples=', len(groups[0]))
    # re-create a new figure each time... there are too many bugs currently in bokeh
    # to support dynamic update of the data
    fig = prepare_new_figure(options, data, data_types)

    # TODO for non-modifying changes, reuse the previous locations (e.g colormap changes)
    """
    if previous_figure is not None:
        print('REUSING RANGE!!')
        # make sure the range of the previous and new picture are sync
        fig.x_range = previous_figure.x_range
        fig.y_range = previous_figure.y_range
        """
    layout_children = [row(fig, sizing_mode='scale_both')] #[fig] #[row(fig, sizing_mode='scale_height')]

    nb_data = utilities.len_batch(data)
    fig.title.text = f'Samples selected: {len(groups[0])}'
    fig.renderers.clear()
    fig.yaxis.visible = False
    fig.xaxis.visible = False

    data_x_name = f'{fig_name}_data_x'
    data_y_name = f'{fig_name}_data_y'

    if data_x_name not in data_source.column_names:
        data_source.add(np.zeros(nb_data, np.float32), data_x_name)
        data_source.add(np.zeros(nb_data, np.float32), data_y_name)
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
        groups, groups_layout_x = scatter(
            data,
            groups,
            scatter_x,
            type_category=type_categories[scatter_x])
        groups = groups[0]
    else:
        # consider all the points as x
        groups_layout_x = [{'value': '', 'type': 'discrete'}]

    if scatter_y is not None:
        groups, groups_layout_y = scatter(
            data,
            groups,
            scatter_y,
            type_category=type_categories[scatter_y])
    else:
        # consider all the points as y
        groups = [[g] for g in groups]
        groups_layout_y = [{'value': '', 'type': 'discrete'}]

    if len(color_by.value) > 0 and color_by.value != 'None':
        color_by = color_by.value
    else:
        color_by = None

    if color_by is not None and type_categories[color_by] == DataCategory.Continuous:
        color_scheme.visible = True
    else:
        color_scheme.visible = False

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

            c = group_coordinate(options, data, group, group_layout_y, group_layout_x)
            data_x[group] = c[:, 0]
            data_y[group] = c[:, 1]

    sync_groups_coordinates(options, data_x, data_y, groups, groups_layout_y, groups_layout_x)

    data_source.data[data_x_name] = data_x
    data_source.data[data_y_name] = data_y

    if color_by is not None:
        # prepare the colors
        if 'discrete' in type_categories[color_by].value:
            colors_all_values = np.asarray(data[color_by], dtype=str)
            color_unique_values = list(set(colors_all_values))
            if options.style.sorted_legend:
                color_unique_values = sorted(color_unique_values)
            color_unique_palette_ints = np.random.randint(0, 255, [len(color_unique_values), 3])
            color_unique_palette_hexs = ['#%0.2X%0.2X%0.2X' % tuple(c) for c in color_unique_palette_ints]

            color_mapper = CategoricalColorMapper(
                factors=color_unique_values,  # must have string
                palette=color_unique_palette_hexs)

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
            sub_fig.text(x=0, y=y_text, text=color_unique_values, x_offset=15, y_offset=6 // 2, text_font_size='6pt')
            layout_children.append(sub_fig)

        elif 'continuous' in type_categories[color_by].value:
            colors_all_values = np.asarray(data[color_by])
            palette = color_scheme.value
            min_value = colors_all_values.min()
            max_value = colors_all_values.max()
            color_mapper = LinearColorMapper(palette=palette, low=min_value, high=max_value)

            bar = ColorBar(
                title=color_by,
                label_standoff=10,
                title_standoff=10,
                color_mapper=color_mapper,
                location=(0, 0),
                ticker=FixedTicker(ticks=np.linspace(min_value, max_value, 5))
                )
            fig.add_layout(bar, "right")

        else:
            raise NotImplementedError()

        data_source.data['color_by'] = colors_all_values
        fill_color = transform('color_by', color_mapper)
    else:
        data_source.data['color_by'] = [None] * nb_data
        fill_color = 'blue'

    if len(icon.value) > 0 and (icon.value != 'Icon' and icon.value != 'Dot'):
        # the images do NOT support tooltips yet in bokeh, instead, overlay a transparent rectangle
        # that will display the tooltip
        units = 'data'
        if 'continuous' in [groups_layout_y[0]['type'], groups_layout_x[0]['type']]:
            # make the size of the image fixed so that when we have
            # many points close by, we can zoom in to isolate the samples
            units = 'screen'

        fig.image_url(url=icon.value, x=data_x_name, y=data_y_name,
                      h=options.image_size, h_units=units,
                      w=options.image_size, w_units=units,
                      anchor="center", source=data_source)
        fig.rect(x=data_x_name, y=data_y_name, width=options.image_size, height=options.image_size, width_units=units, height_units=units, source=data_source, fill_alpha=0, line_alpha=0)
    
        if color_by is not None:
            fig.rect(
                source=data_source,
                x=data_x_name,
                y=data_y_name,
                width=options.image_size - options.style.color_by_line_width,  # TODO data space mixed with screen space
                height=options.image_size - options.style.color_by_line_width,
                line_color=fill_color,
                fill_color=None,
                line_width=options.style.color_by_line_width,
                width_units=units, height_units=units,
            )

    elif icon.value == 'Icon':
        fig.oval(x=data_x_name, y=data_y_name, width=options.image_size, height=options.image_size, width_units='data', height_units='data', source=data_source, fill_color=fill_color)
    elif icon.value == 'Dot':
        fig.circle(x=data_x_name, y=data_y_name, size=5, source=data_source, line_color='black', fill_color=fill_color)
    else:
        raise NotImplementedError()

    if scatter_x and groups_layout_x[0]['type'] == 'discrete':
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
        fig.xaxis.major_label_overrides = {t: str(name) for t, name in zip(ticks, ticks_label)}
        fig.xaxis.major_label_orientation = math.pi / 4
    elif scatter_x and groups_layout_x[0]['type'] == 'continuous':
        fig.xaxis.visible = True
        ticks = [groups_layout_x[0]['coord_min'], groups_layout_x[0]['coord_max']]
        ticks_label = [groups_layout_x[0]['value_min'], groups_layout_x[0]['value_max']]
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = {t: str(name) for t, name in zip(ticks, ticks_label)}
    else:
        fig.xaxis.visible = False

    if scatter_y and groups_layout_y[0]['type'] == 'discrete':
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
        fig.yaxis.major_label_overrides = {t: str(name) for t, name in zip(ticks, ticks_label)}
        fig.yaxis.major_label_orientation = math.pi / 4
    elif scatter_y and groups_layout_y[0]['type'] == 'continuous':
        fig.yaxis.visible = True
        ticks = [groups_layout_y[0]['coord_min'], groups_layout_y[0]['coord_max']]
        ticks_label = [groups_layout_y[0]['value_min'], groups_layout_y[0]['value_max']]
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = {t: str(name) for t, name in zip(ticks, ticks_label)}
    else:
        fig.yaxis.visible = False

    print('render command DONE')
    return layout_children


def make_custom_tooltip(options, data, data_types):
    tips = []
    for key in data.keys():
        # create a custom tooltip to display images
        t = data_types.get(key)
        if t is not None and 'BLOB_IMAGE' in t:
            tip = f"""
            <div style="opacity: 1.0;">
                <img
                    src="@{key}" width="100%"
                    border="2"
                </img>
                <span>{key}</span>
            </div>
            """
            tips.append(tip)
        else:
            tips.append(f'<div><span style="">{key}: @{key}</span></div>')

    # this style make sure that a single tooltip is displayed at once. If
    # many samples overlap, it may be really slow to render these
    # and they will be out of the screen anyway
    div_style = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>
    """
    return '<div>' + div_style + '\n'.join(tips) + '</div>'


def prepare_new_figure(options, data, data_types):
    tools = 'pan,wheel_zoom,reset'
    f = figure(
        title='',
        tools=tools,
        active_scroll='wheel_zoom',
        match_aspect=True,
        aspect_ratio=options.style.scatter_aspect_ratio,
        aspect_scale=1,
        toolbar_location='above',
    )

    hover_tool = HoverTool(tooltips=make_custom_tooltip(options, data, data_types))
    f.add_tools(hover_tool)

    f.xgrid.visible = False
    f.ygrid.visible = False
    return f


def process_data_samples__scatter(options, name, data, data_types, type_categories):
    print('process_data_samples__scatter')
    values = ['None'] + list(sorted(data.keys()))

    default_scatter_x_axis = safe_lookup(options.config, name, 'default', 'Scatter X Axis', default='None')
    scatter_x_axis = Select(title='Scatter X Axis', options=values, value=default_scatter_x_axis)
    default_scatter_y_axis = safe_lookup(options.config, name, 'default', 'Scatter Y Axis', default='None')
    scatter_y_axis = Select(title='Scatter Y Axis', options=values, value=default_scatter_y_axis)
    default_color_by = safe_lookup(options.config, name, 'default', 'Color by', default='None')
    color_by = Select(title='Color by', options=values, value=default_color_by)
    default_color_scheme = safe_lookup(options.config, name, 'default', 'Color scheme', default='Viridis256')
    color_scheme = Select(title='Color scheme', options=['Viridis256', 'Magma256', 'Greys256', 'Turbo256'], value=default_color_scheme)
    default_binning_x_axis = safe_lookup(options.config, name, 'default', 'Binning X Axis', default='None')
    binning_x_axis = Select(title='Binning X Axis', options=values, value=default_binning_x_axis)
    default_label_with = safe_lookup(options.config, name, 'default', 'Label with', default='None')
    label_with = Select(title='Label with', options=values, value=default_label_with)

    default_icon = safe_lookup(options.config, name, 'default', 'Display with', default='None')
    icon_values = [v for v, t in data_types.items() if 'BLOB_IMAGE' in t] + ['Icon', 'Dot']
    if default_icon != 'None':
        if default_icon in icon_values:
            del icon_values[icon_values.index(default_icon)]
            icon_values.insert(0, default_icon)
    icon = Select(title='Display with', options=icon_values, value=icon_values[0])

    controls_list = [
        scatter_x_axis,
        scatter_y_axis,
        color_by,
        color_scheme,
        binning_x_axis,
        icon,
        label_with
    ]
    controls = column(controls_list, width=options.style.tool_window_size_x, height=options.style.tool_window_size_y)
    controls.sizing_mode = "fixed"

    layout = row(controls, sizing_mode='scale_height')

    update = functools.partial(
        render_data,
        options=options,
        data=data,
        data_types=data_types,
        type_categories=type_categories,
        scatter_x=scatter_x_axis,
        scatter_y=scatter_y_axis,
        binning_x_axis=binning_x_axis,
        color_by=color_by,
        color_scheme=color_scheme,
        icon=icon,
        label_with=label_with,
        layout=layout,
    )

    for control in controls_list:
       control.on_change('value', lambda attr, old, new: update())

    update()

    return Panel(child=layout, title='Scatter')


class DataCategory(Enum):
    Continuous = 'continuous'
    DiscreteOrdered = 'discrete_ordered'
    DiscreteUnordered = 'discrete_unordered'
    Other = 'other'

    @staticmethod
    def from_numpy_array(array):
        if len(array.shape) == 1:
            if np.issubdtype(array.dtype, np.integer):
                return DataCategory.DiscreteOrdered
            elif np.issubdtype(array.dtype, np.floating):
                return DataCategory.Continuous
            elif np.issubdtype(array.dtype, np.str):
                return DataCategory.DiscreteUnordered

        return DataCategory.Other


def normalize_data(options, data, table_name):
    """
    Normalize, subsample and categorize the data

    The following operations are performed:
    - convert to numpy arrays
    - removed type column and return a specific `type` dictionary
    - normalize the path according to deployment: if static html, nothing to do
        if deployed, we MUST add the `app name` (the top folder name containing the SQL DB)
        as root
    - categorize the data as continuous, discrete unordered, discrete ordered, other
    - recover the DB type (string) from data values (e.g., float or int)
    """
    d = collections.OrderedDict()
    for name, values in data.items():
        d[name] = np.asarray(values)
    types = get_data_types_and_clean_data(d)
    type_categories = {}

    # handle the column removal: maybe they are not useful or maybe the data
    # can't be parsed
    remove_columns = safe_lookup(options.config, table_name, 'data', 'remove_columns', default=[])
    if remove_columns is not None:
        for n in remove_columns:
            if n in data:
                del d[n]
            if n in types:
                del types[n]

    subsampling_factor = safe_lookup(
        options.config,
        table_name,
        'data',
        'subsampling_factor',
        default='1.0')
    subsampling_factor = float(subsampling_factor)
    assert subsampling_factor <= 1.0, 'sub-sampling factor must be <= 1.0'
    if subsampling_factor != 1.0:
        nb_samples = int(len_batch(d) * subsampling_factor)
        d = {
            name: values[:nb_samples] for name, values in d.items()
        }

    if options.embedded:
        appname = os.path.basename(os.path.dirname(options.db_root))
        for n, t in types.items():
            if 'BLOB_' in t:
                # it is a blob, meaning that it was saved on the local HD
                d[n] = list(np.core.defchararray.add(np.asarray([f'{appname}/']), d[n]))

            if 'BLOB_IMAGE' in t:
                type_categories[n] = DataCategory.Other

    # load the numpy arrays
    for name, t in list(types.items()):
        if 'BLOB_NUMPY' in t:
            loaded_np = []
            root = os.path.join(os.path.dirname(options.db_root), '..')
            for relative_path in d[name]:
                path = os.path.join(root, relative_path)
                loaded_np.append(np.load(path))

            # expand the array if satisfying the criteria
            array = np.asarray(loaded_np)
            if len(array.shape) == 2 and array.shape[1] <= options.data.unpack_numpy_arrays_with_less_than_x_columns:
                for n in range(array.shape[1]):
                    name_expanded = name + f'_{n}'
                    value_expanded = array[:, n]
                    d[name_expanded] = value_expanded
                    type_categories[name_expanded] = DataCategory.from_numpy_array(value_expanded)
                del d[name]
                del types[name]
            else:
                d[name] = loaded_np
                type_categories[name] = DataCategory.Other  # dimension too high

    # the DB converted all values to string and have lost the original type. Try to
    # revert back the type from the column values
    for name in list(d.keys()):
        t = d[name]
        if name not in type_categories:
            try:
                t_np = np.asarray(t, dtype=np.int)  # if e have a float, it will raise exception
                type_categories[name] = DataCategory.DiscreteOrdered
                d[name] = list(t_np)
                continue  # success, go to next item
            except ValueError:
                pass

            try:
                t_np = np.asarray(t, dtype=np.float32)
                type_categories[name] = DataCategory.Continuous
                d[name] = list(t_np)
                continue  # success, go to next item
            except ValueError:
                type_categories[name] = DataCategory.Other

            try:
                t_np = np.asarray(t, dtype=np.str)
                type_categories[name] = DataCategory.DiscreteUnordered
                continue  # success, go to next item
            except ValueError:
                type_categories[name] = DataCategory.Other

    return d, types, type_categories


def process_data_samples(options, connection, name):
    """

    Args:
        options:
        connection:
        name:

    Returns:

    """
    data, types, type_categories = normalize_data(options, get_table_data(connection, name), table_name=name)

    tabs = []
    if options.data_samples.display_tabular:
        tabs.append(process_data_samples__tabular(options, data, types, type_categories))

    if options.data_samples.display_scatter and options.embedded:
        # here we require some python logic, so we need to have a bokeh
        # server running to display this view
        tabs.append(process_data_samples__scatter(options, name, data, types, type_categories))

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


def run_server(path_to_db, options=create_default_reporting_options(embedded=True)):
    """
    Run a server

    Note:
        the .db will be replaced by .json if a file with this name is present,
        it will be imported and used as configuration

    Args:
        path_to_db: the path to the SQLite database
        options: configuration options

    Returns:

    """
    def fn(doc):
        return report(sql_database_path=path_to_db, options=options, doc=doc)

    app_1 = Application(FunctionHandler(fn))

    if len(options.config) == 0:
        json_config = path_to_db.replace('.db', '.json')
        if os.path.exists(json_config):
            with open(json_config, 'r') as f:
                config = json.load(f)
                options.config = config

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

    config = {
        'samples': {
            'data': {
                'remove_columns': [
                    'sex',
                    #'images'
                ],
                'subsampling_factor': 1.0,
            },
            'default': {
                #'Scatter X Axis': 'diagnosis',
                #'Scatter Y Axis': 'term_classification_output_raw_1',
                #'Color by': 'term_classification_output',
                #'Color by': 'term_classification_output',
                #'Display with': 'Dot'
            }
        }
    }

    options = create_default_reporting_options(config=config)
    options.image_size = 64
    run_server('C:/trw_logs/mnist_cnn_r0/reporting_sqlite.db', options=options)F

    # run a static HTML page (i.e., the more complicated views requiring python v=callbacks will be disabled)
    # report('/path/reporting_sqlite.db', options=create_default_reporting_options(embedded=False))
