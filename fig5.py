import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian
from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import minimize_scalar
from functions import init_fig, simpleaxis
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper

if matplotlib.__version__[0] == '2':
    matplotlib.style.use('classic')

try:
    from sys import argv
    from os.path import isdir
    figpath = argv[1] if isdir(argv[1]) else False
except:
    figpath = False

init_fig()

# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
[vermillon, orange, yellow, green, cyan, blue, purple, grey] = col

#

# Load results

A2 = np.load('results/CNMF-HRshapes.npz')['A2'].item()
N = A2.shape[1]
b2 = np.load('results/CNMF-HRshapes.npz')['b2']
ssub = np.load('results/decimate.npz')['ssub'].item()
srt = cse.utilities.order_components(A2, ssub[1][0])[-1]
ssubLR = np.load('results/decimate-LR.npz')['ssub'].item()
A2LR = np.load('results/CNMF-LRshapes.npz')['A2'].item()
b2LR = np.load('results/CNMF-LRshapes.npz')['b2']

#

# plot traces

dsls = [1, 4, 16]
fps = 20
ticksep = 40
shapes = A2.T.toarray().reshape(-1, 512, 512)
N, T = ssub[1][0].shape


def GetBox(centers, R, dims):
    D = len(R)
    box = np.zeros((D, 2), dtype=int)
    for dd in range(D):
        box[dd, 0] = max((centers[dd] - R[dd], 0))
        box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
    return box


def baseline(s, f, b, bl):
    return ((f.ravel() * s.ravel().dot(b.ravel())) / s.ravel().dot(s.ravel()) + bl)


def df(a, s, f, b, bl):
    return (a - bl) / baseline(s, f, b, bl)


boxes = [GetBox(center_of_mass(s), [14, 14], [512, 512]) for s in shapes]
fig = plt.figure(figsize=(25, 15))
for k, neuronId in enumerate(srt[np.array([0, 39, 79, 119])]):
    ax0 = fig.add_axes([.023, .775 - .244 * k, .973, .223])
    mi = np.inf
    for i, ds in enumerate(dsls):
        l, = plt.plot(df(ssub[ds][0][neuronId], shapes[neuronId], ssub[ds][1],
                         b2, ssub[ds][3][neuronId]), ['-', '-', '--'][i],
                      label='%dx%d' % (ds, ds), alpha=1., clip_on=False, zorder=10,
                      lw=3, c=[green, yellow, vermillon][i])
        if i == 1:
            l.set_dashes([14, 10])
    foo = df(ssub[1][0][neuronId], shapes[neuronId], ssub[1][1], b2, ssub[1][3][neuronId])
    bar = df(ssubLR[4][0][neuronId], A2LR.T.toarray()[neuronId], ssubLR[4][1],
             b2LR, ssubLR[4][3][neuronId])
    z = minimize_scalar(lambda x: np.sum((foo - x * bar)**2))['x']
    plt.plot(z * bar, clip_on=False, zorder=-10, lw=3, c=cyan)
    lb, ub = plt.ylim()
    for i, ds in enumerate(dsls):
        scale = baseline(shapes[neuronId], ssub[ds][1], b2, ssub[ds][3][neuronId]).mean()
        l, = plt.plot(ssub[ds][2][neuronId] / scale - .7 * ub,
                      ['-', '-', '--'][i], alpha=1., c=[green, yellow, vermillon][i])
    if i == 1:
        l.set_dashes([14, 10])
    plt.plot(z * ssubLR[4][2][neuronId] / scale - .7 * ub, alpha=1., c=cyan, zorder=-10)
    plt.ylim(-.7 * ub, ub)
    plt.legend(frameon=False, ncol=len(dsls), loc=[.073, .775], columnspacing=14.2)
    plt.xticks(range(0, T, ticksep * fps), ['', '', ''])
    tmp = [500, 300, 100, 100][k]
    plt.plot([0, 0], [0, 1], c='k', lw=5, clip_on=False, zorder=11)
    plt.text(7, 1, '100\%', verticalalignment='center')
    simpleaxis(plt.gca())
    plt.gca().spines['left'].set_visible(False)
    plt.yticks([])
    for i, ds in enumerate(dsls):
        ax = fig.add_axes([[.17, .467, .78][i], .838 - .244 * k, .035, .2625])
        ss = downscale_local_mean(
            gaussian(shapes[neuronId], 1), (ds, ds)).repeat(ds, 0).repeat(ds, 1)
        ax.imshow(ss[map(lambda a: slice(*a), boxes[neuronId])],
                  cmap='hot', interpolation='nearest')
        ax.axis('off')
        ax.text(1.25, .5, '%.3f' % (np.corrcoef(ssub[ds][0][neuronId],
                                                ssub[1][0][neuronId])[0, 1]),
                verticalalignment='center', transform=ax.transAxes)
ax0.set_xticklabels(range(0, T / fps, ticksep))
ax0.set_xlabel('Time [s]', labelpad=-15)
ax0.set_ylabel('Activity and $\Delta$F/F', labelpad=10, y=1.5)
plt.savefig(figpath + '/traces.pdf') if figpath else plt.show(block=True)
