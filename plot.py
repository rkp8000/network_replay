from aux import load_time_file
import numbers
import shelve
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def set_font_size(ax, font_size, legend_font_size=None):
    """Set font_size of all axis text objects to specified value."""

    texts = [ax.title, ax.xaxis.label, ax.yaxis.label] + \
        ax.get_xticklabels() + ax.get_yticklabels()

    for text in texts:
        text.set_fontsize(font_size)

    if ax.get_legend():
        if not legend_font_size:
            legend_font_size = font_size
        for text in ax.get_legend().get_texts():
            text.set_fontsize(legend_font_size)


def raster(ax, time_file, activity_file, order=None, colors='k'):
    """
    Make a raster plot of spiking activity from a ntwk simulation.

    :param ax: axis object
    :param time_file: path to file containing timestamps and sampling freq
    :param ntwk_file: path to file containing ntwk activity
    :param order: ordering of neurons
    """

    # load timestamps
    ts, fs = load_time_file(time_file)

    # load activity
    data = shelve.open(activity_file)

    # check arguments
    if order is None:
        spks = data['spks']
        order = range(spks.shape[1])
    else:
        spks = data['spks'][:, order]

    if len(colors) > 1 and not isinstance(colors[0], numbers.Number):
        # i.e., if colors is a sequence of colors
        if len(colors) != len(order):
            raise ValueError('"colors" must have same length as "order".')
    else:
        colors = [colors] * len(order)

    # plot spks
    spk_times, spk_rows = spks.nonzero()
    spk_times = spk_times/fs

    cs = [colors[cell] for cell in spk_rows]

    ax.scatter(spk_times, spk_rows, c=cs, lw=1, marker='|')

    ax.set_xlabel('t (s)')
    ax.set_ylabel('neuron')
