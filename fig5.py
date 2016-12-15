import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian
from scipy.ndimage.measurements import center_of_mass
from functions import init_fig, simpleaxis
import ca_source_extraction as cse


init_fig()
save_figs = False  # subfolder fig must exist if set to True


# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
[vermillon, orange, yellow, green, cyan, blue, purple, grey] = col

#

# Load results

A2 = np.load('results/CNMF-HRshapes.npz')['A2'].item()
N = A2.shape[1]
ssub = np.load('results/decimate.npz')['ssub'].item()
srt = cse.utilities.order_components(A2, ssub[1][0])[-1]

#

# plot traces

dsls = [1, 4, 16]
fps = 20
ticksep = 40
shapes = A2.T.toarray().reshape(-1, 512, 512)
shapes = np.asarray([gaussian(a, 1) for a in shapes])
N, T = ssub[1][0].shape


def GetBox(centers, R, dims):
    D = len(R)
    box = np.zeros((D, 2), dtype=int)
    for dd in range(D):
        box[dd, 0] = max((centers[dd] - R[dd], 0))
        box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
    return box


boxes = [GetBox(center_of_mass(s), [24, 24], [512, 512]) for s in shapes]
fig = plt.figure(figsize=(25, 15))
for k, neuronId in enumerate(srt[np.array([0, 39, 79, 119])]):
    ax0 = fig.add_axes([.04, .763 - .24 * k, .957, .223])
    mi = np.inf
    for i, ds in enumerate(dsls):
        l, = plt.plot(ssub[ds][0][neuronId], ['-', '-', '--'][i], label='%dx%d' % (ds, ds),
                      alpha=1., clip_on=False, zorder=10, lw=3, c=[green, cyan, orange][i])
        if i == 1:
            l.set_dashes([14, 10])
        mi0 = np.min(ssub[ds][0][neuronId])
        if mi0 < mi:
            mi = mi0
    lb, ub = plt.ylim()
    maS = 0
    for ds in dsls:
        maS0 = np.max(ssub[ds][2][neuronId])
        if maS0 > maS:
            maS = maS0
    for i, ds in enumerate(dsls):
        plt.plot(2 * ssub[ds][2][neuronId] + mi - 2.2 *
                 maS, alpha=1., c=[green, cyan, orange][i])
    plt.ylim(mi - 2.2 * maS, ub)
    plt.legend(frameon=False, ncol=len(dsls),
               loc=[.068, .81], columnspacing=12.9)
    plt.xticks(range(0, T, ticksep * fps), ['', '', ''])
    plt.yticks([0, 1000 * int(plt.ylim()[1]) / 1000],
               [0, int(plt.ylim()[1]) / 1000])
    simpleaxis(plt.gca())
    for i, ds in enumerate(dsls):
        ax = fig.add_axes([[.185, .46, .75][i], .835 - .24 * k, .035, .2625])
        ss = downscale_local_mean(
            shapes[neuronId], (ds, ds)).repeat(ds, 0).repeat(ds, 1)
        ax.imshow(ss[map(lambda a: slice(*a), boxes[neuronId])],
                  cmap='hot', interpolation='nearest')
        ax.axis('off')
        ax.text(1.3, .5, '%.3f' % (np.corrcoef(ssub[ds][0][neuronId], ssub[1][0][neuronId])[0, 1]),
                verticalalignment='center', transform=ax.transAxes)
ax0.set_xticklabels(range(0, T / fps, ticksep))
ax0.set_xlabel('Time [s]', labelpad=-15)
ax0.set_ylabel('Activity and Fluorescence [a.u.]', labelpad=5, y=2.05)
if save_figs:
    plt.savefig('fig/traces.pdf')
plt.show()
