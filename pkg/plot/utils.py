import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patheffects import Normal, Stroke
from matplotlib.transforms import Bbox


def shrink_axis(ax, scale=0.7):
    pos = ax.get_position()
    mid = (pos.ymax + pos.ymin) / 2
    height = pos.ymax - pos.ymin
    new_pos = Bbox(
        [
            [pos.xmin, mid - scale * 0.5 * height],
            [pos.xmax, mid + scale * 0.5 * height],
        ]
    )
    ax.set_position(new_pos)


def draw_colors(ax, ax_type="x", labels=None, palette="tab10"):
    r"""
    Draw colormap onto the axis to separate the data
    Parameters
    ----------
    ax : matplotlib axes object
        Axes in which to draw the colormap
    ax_type : char, optional
        Setting either the x or y axis, by default "x"
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    divider : AxesLocator, optional
        Divider used to add new axes to the plot
    color : str, list of str, or array_like, optional
        Attribute in meta by which to draw colorbars, by default None
    palette : str or dict, optional
        Colormap of the colorbar, by default "tab10"
    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the color map
    """

    uni_classes = np.unique(labels)
    # Create the color dictionary
    color_dict = palette

    # Make the colormap
    class_map = dict(zip(uni_classes, range(len(uni_classes))))
    color_sorted = np.vectorize(color_dict.get)(uni_classes)
    color_sorted = np.array(color_sorted).T
    lc = ListedColormap(color_sorted)
    class_indicator = np.vectorize(class_map.get)(labels)

    if ax_type == "x":
        class_indicator = class_indicator.reshape(1, len(labels))
    elif ax_type == "y":
        class_indicator = class_indicator.reshape(len(labels), 1)
    sns.heatmap(
        class_indicator,
        cmap=lc,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
        ax=ax,
        square=False,
    )
    return ax


def remove_shared_ax(ax):
    """
    Remove ax from its sharex and sharey
    """
    # Remove ax from the Grouper object
    shax = ax.get_shared_x_axes()
    shay = ax.get_shared_y_axes()
    shax.remove(ax)
    shay.remove(ax)

    # Set a new ticker with the respective new locator and formatter
    for axis in [ax.xaxis, ax.yaxis]:
        ticker = mpl.axis.Ticker()
        axis.major = ticker
        axis.minor = ticker
        # No ticks and no labels
        loc = mpl.ticker.NullLocator()
        fmt = mpl.ticker.NullFormatter()
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(fmt)


def make_sequential_colormap(cmap="Blues", vmin=0, vmax=1):
    # REF: basically taken from seaborn somewhere
    cmap = mpl.cm.get_cmap(cmap)
    normlize = mpl.colors.Normalize(vmin, vmax)
    cmin, cmax = normlize([vmin, vmax])
    cc = np.linspace(cmin, cmax, 256)
    cmap = mpl.colors.ListedColormap(cmap(cc))
    return cmap


def get_text_points(text, transformer, renderer):
    bbox = text.get_window_extent(renderer=renderer)
    bbox_points = bbox.get_points()
    out_points = transformer.transform(bbox_points)
    return out_points


def get_text_width(text, transformer, renderer):
    points = get_text_points(text, transformer, renderer)
    width = points[1][0] - points[0][0]
    return width


def multicolor_text(x, y, texts, colors, ax=None, space_scale=1.0, **kwargs):
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    transformer = ax.transData.inverted()

    # make this dummy text to get proper space width, then delete
    text = ax.text(0.5, 0.5, " ")
    space_width = get_text_width(text, transformer, renderer)
    space_width *= space_scale
    text.remove()

    # TODO make the spacing "smart"
    text_objs = []
    for text, color in zip(texts, colors):
        text_obj = ax.text(x, y, text, color=color, **kwargs)
        text_width = get_text_width(text_obj, transformer, renderer)
        x += text_width
        x += space_width
        text_objs.append(text_obj)

    return text_objs


def get_texts_points(texts, ax=None):
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    transformer = ax.transData.inverted()

    x_maxs = []
    x_mins = []
    y_maxs = []
    y_mins = []
    for text in texts:
        points = get_text_points(text, transformer, renderer)
        x_maxs.append(points[1][0])
        x_mins.append(points[0][0])
        y_maxs.append(points[1][1])
        y_mins.append(points[0][1])

    x_max = max(x_maxs)
    x_min = min(x_mins)
    y_max = max(y_maxs)
    y_min = min(y_mins)
    return x_min, x_max, y_min, y_max


def bound_texts(texts, ax=None, xpad=0, ypad=0, **kwargs):
    x_min, x_max, y_min, y_max = get_texts_points(texts, ax=ax)
    xy = (x_min - xpad, y_min - ypad)
    width = x_max - x_min + 2 * xpad
    height = y_max - y_min + 2 * ypad
    patch = mpl.patches.Rectangle(xy=xy, width=width, height=height, **kwargs)
    ax.add_patch(patch)


def nice_text(
    x,
    y,
    s,
    ax=None,
    color="black",
    fontsize=None,
    transform=None,
    ha="left",
    va="center",
    linewidth=4,
    linecolor="black",
):
    if transform is None:
        transform = ax.transData
    text = ax.text(
        x,
        y,
        s,
        color=color,
        fontsize=fontsize,
        transform=transform,
        ha=ha,
        va=va,
    )
    text.set_path_effects([Stroke(linewidth=linewidth, foreground=linecolor), Normal()])


