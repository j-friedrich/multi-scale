import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def init_fig():
    plt.rc('figure', figsize=(8, 6), facecolor='white', dpi=90, frameon=False)
    plt.rc('font', size=30, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
    plt.rc('lines', lw=2)
    plt.rc('text', usetex=True)
    plt.rc('legend', **{'fontsize': 30, 'frameon': False, 'labelspacing': .3, 'handletextpad': .3})
    plt.rc('axes', linewidth=2)
    plt.rc('xtick.major', size=10, width=1.5)
    plt.rc('ytick.major', size=10, width=1.5)
    plt.rc('hatch', linewidth=1.2)
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


def IQRfill(cor, dsls, c="#E69F00", hatch=None, ls='-'):
    y1 = np.percentile(cor, 75, 0)
    y2 = np.percentile(cor, 25, 0)
    plt.plot(dsls[:cor.shape[1]], y1, ls=ls, lw=2, c=c)
    plt.plot(dsls[:cor.shape[1]], y2, ls=ls, lw=2, c=c)
    plt.fill_between(dsls[:cor.shape[1]], y1, y2, facecolor=c if hatch is None else 'none',
                     edgecolor=c, alpha=.3 if hatch is None else 1, hatch=hatch, lw=0,
                     zorder=-11 if hatch is not None else 0)


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
