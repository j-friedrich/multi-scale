import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CNMF import LocalNMF, HALS4activity
from functions import init_fig, simpleaxis, noclip
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper

init_fig()
save_figs = False  # subfolder fig must exist if set to True

# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
[vermillon, orange, yellow, green, cyan, blue, purple, grey] = col

cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
gfp = matplotlib.colors.LinearSegmentedColormap('GFP_colormap', cdict, 256)


# Fetch Data
data = np.load('data_zebrafish.npy')
sig = (3, 3)
# find neurons greedily
N = 43
centers = cse.greedyROI(data.reshape((-1, 30) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[4, 4])[2]


# get shapes on data with high spatial resolution
MSE_array, shapes, activity, boxes = LocalNMF(
    data, centers, sig, iters=5, mb=20, iters0=30, ds=[3, 3])
# scale
z = np.sqrt(np.sum(shapes.reshape(len(shapes), -1)[:-1]**2, 1)).reshape(-1, 1)
activity[:-1] *= z
shapes[:-1] /= z.reshape(-1, 1, 1)

# decimate
activityDS = {}
dsls = [1, 2, 3, 4, 6, 8, 12]
for ds in dsls:
    activityDS[ds] = HALS4activity(data.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                                   .mean(-1).mean(-2).reshape(len(data), -1),
                                   shapes.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                                   .mean(-1).mean(-2).reshape(len(shapes), -1),
                                   activity.copy(), 10)

# plot traces
fig = plt.figure(figsize=(17, 6))
for k, neuronId in enumerate([1, 5, 27]):
    ax0 = fig.add_axes([.055, .71 - .3 * k, .925, .28])
    for i, ds in enumerate([1, 2, 4, 8]):
        l, = plt.plot(activityDS[ds][neuronId][1200:2400], ['-', '-', '--', '-.'][i],
                      c=[green, cyan, orange, 'k'][i], label='%dx%d' % (ds, ds))
        if i == 1:
            l.set_dashes([14, 10])
    plt.legend(frameon=False, ncol=4, loc=[.073, .65], columnspacing=5.63, handletextpad=.1)
    plt.xticks(range(0, 2000, 1200), ['', '', ''])
    plt.ylim(0, [330, 1330, 360][k])
    tmp = [2, 10, 2][k]  # int(activityDS[ds][neuronId].max() / 100)-2
    plt.yticks([0, 100 * tmp], [0, tmp])
    simpleaxis(plt.gca())
    for i, ds in enumerate([1, 2, 4, 8]):
        ax = fig.add_axes([.22 + i * .225, .82 - .3 * k, .04, .24])
        ss = (shapes[neuronId].reshape(shapes.shape[1] / ds, ds, shapes.shape[2] / ds, ds)
              .mean(-1).mean(-2).repeat(ds, 0).repeat(ds, 1))
        ax.imshow(ss[map(lambda a: slice(*a), boxes[neuronId])][::-1],
                  cmap='hot', interpolation='nearest')
        ax.axis('off')
        ax.text(1.2, .6, '%.3f' % (np.corrcoef(activityDS[ds][neuronId],
                                               activityDS[1][neuronId])[0, 1]),
                verticalalignment='center', transform=ax.transAxes)
ax0.set_xticklabels([10, 20])
# http://stackoverflow.com/questions/28615887/how-to-move-a-ticks-label-in-matplotlib
import types
SHIFT = -10.  # Data coordinates
for label in ax0.xaxis.get_majorticklabels()[:1]:
    label.customShiftValue = SHIFT
    label.set_x = types.MethodType(
        lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
        label, matplotlib.text.Text)
ax0.set_xlabel('Time [min]', labelpad=-15)
ax0.set_ylabel('Fluorescence [a.u.]', y=1.55, labelpad=15)
if save_figs:
    plt.savefig('fig/traces_zebrafish.pdf')
plt.show()


# get shapes not on high-res but coarse data
activityDSlr = {}
activityDSlr[1] = activityDS[1]
shapesDSlr = {}
shapesDSlr[1] = shapes
for ds in dsls[1:]:
    MSE_array, shapesLR, activity, boxes = LocalNMF(
        data.reshape(-1, 96 / ds, ds, 96 / ds, ds).mean(-1).mean(-2),
        1. * centers / ds, 1. * np.array(sig) / ds, iters=5,
        mb=20, iters0=30, ds=[1 + 2 / ds, 1 + 2 / ds])
    activityDSlr[ds] = HALS4activity(
        data.reshape(-1, 96 / ds, ds, 96 / ds, ds)
        .mean(-1).mean(-2).reshape(len(data), -1),
        shapesLR.reshape(len(shapesLR), -1), activity.copy(), 10)
    shapesDSlr[ds] = shapesLR


# plot correlation values


def foo(ssub, comp, idx=None):
    N, T = comp.shape
    if idx is None:
        idx = range(N)
    cor = np.zeros((len(idx), len(dsls))) * np.nan
    for i, ds in enumerate(dsls):
        cor[:, i] = np.array(
            [np.corrcoef(ssub[ds][n], comp[n])[0, 1] for n in idx])
    return cor

fig = plt.figure(figsize=(6.5, 6))
cor = foo(activityDSlr, activityDS[1])
plt.plot(dsls[:6], np.mean(cor, 0)[:6], lw=4, c=cyan,
         label='1 phase\n imaging', clip_on=False)
noclip(plt.errorbar(dsls[:6], np.mean(cor, 0)[:6],
                    yerr=np.std(cor, 0)[:6] / np.sqrt(len(cor)),
                    lw=3, capthick=2, fmt='o', c=cyan, clip_on=False))
cor = foo(activityDS, activityDS[1])
plt.plot(dsls, np.mean(cor, 0), lw=4, c=orange,
         label='2 phase\n imaging', clip_on=False)
noclip(plt.errorbar(dsls, np.mean(cor, 0),
                    yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                    lw=3, capthick=2, fmt='o', c=orange, clip_on=False))
plt.xlabel('Spatial decimation')
simpleaxis(plt.gca())
plt.xticks(dsls, ['1x1', '', '', '4x4', '6x6', '8x8', '12x12'])
plt.yticks(*[np.round(np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.1), 1)] * 2)
plt.xlim(dsls[0], dsls[-1])
plt.ylabel('Correlation')
plt.legend(loc=(.01, .01), ncol=1)
plt.subplots_adjust(.155, .15, .925, .96)
if save_figs:
    plt.savefig('fig/Corr_zebrafish.pdf')
plt.show()
