import matplotlib
import matplotlib.pyplot as plt


def init_fig():
    plt.rc('figure', figsize=(8, 6), facecolor='white', dpi=90, frameon=False)
    plt.rc('font', size=30, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
    plt.rc('lines', lw=2)
    plt.rc('text', usetex=True)
    plt.rc('legend', **{'fontsize': 30, 'frameon': False, 'labelspacing': .3, 'handletextpad': .3})
    plt.rc('axes', linewidth=2)
    plt.rc('xtick.major', size=10, width=1.5)
    plt.rc('ytick.major', size=10, width=1.5)
    plt.ion()


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def noclip(e):
    for b in e[1]:
        b.set_clip_on(False)
    for b in e[2]:
        b.set_clip_on(False)


def showpause(t=2):
    plt.show()
    plt.pause(t)


# define colormaps
cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
gfp = matplotlib.colors.LinearSegmentedColormap('GFP_colormap', cdict, 256)

cdict2 = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'blue': ((0.0, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))}
blue2green = matplotlib.colors.LinearSegmentedColormap('Blue2Green_colormap', cdict2, 256)
