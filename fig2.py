import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from CNMF import LocalNMF
from functions import init_fig, simpleaxis, gfp, showpause
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper

try:
    from sys import argv
    from os.path import isdir
    figpath = argv[1] if isdir(argv[1]) else False
except:
    figpath = False

init_fig()

# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
# vermillon, orange, yellow, green, cyan, blue, purple, grey


# load data
data = np.load('data_zebrafish.npy')
sig = (3, 3)
# find neurons greedily
N = 49
centers = cse.initialization.initialize_components(data.transpose(1, 2, 0), N)[-1]


# run NMF
MSE_array, shapes, activity, boxes = LocalNMF(data, centers, sig, verbose=True,
                                              iters=100, iters0=30, mb=30, ds=[3, 3])
MSE_array0, shapes0, activity0, boxes0 = LocalNMF(data, centers, sig, verbose=True,
                                                  iters=100, iters0=0, mb=0)
MSE_arrayQ, shapesQ, activityQ, boxesQ = LocalNMF(data, centers, sig, verbose=True,
                                                  iters=5, iters0=30, mb=30, ds=[3, 3])


# plot patch
idx = [0, 14, 36]
np.random.seed(0)
fig = plt.figure(figsize=(3, 3), frameon=False)
plt.imshow(np.sqrt(data.reshape((-1, 10) + data.shape[1:]).mean(1).max(0)).T, gfp)
for b in boxes:
    plt.gca().add_patch(plt.Rectangle(
        b[:, 0] + .2, b[0, 1] - b[0, 0] - 1, b[1, 1] - b[1, 0] - 1,
        linestyle='dashed', lw=1.5, fill=False,
        ec=np.array([.2, .2, .2]) + np.array([.8, .5, .8]) * np.random.rand(3)))
for i, k in enumerate(idx):
    plt.gca().add_patch(plt.Rectangle(boxes[k][:, 0] + .2,
                                      boxes[k][0, 1] - boxes[k][0, 0] - 1,
                                      boxes[k][1, 1] - boxes[k][1, 0] - 1,
                                      lw=2, fill=False, ec=col[2 * i]))
    plt.scatter(*centers[k], s=60, marker='x', lw=3, c=col[2 * i], zorder=10)
plt.axis('off')
plt.xlim(0, 96)
plt.ylim(96, 0)
plt.subplots_adjust(0, 0, 1, 1)
plt.savefig(figpath + '/patch.pdf') if figpath else showpause()


# plot series for short and long processing time
for j, a in enumerate([activityQ, activity]):
    plt.figure(figsize=(18, 6. / 8 * 7))
    for i, k in enumerate(idx):
        tmp = 1.1 * (2 - i) + activity0[k, :770] / activity0[k, :770].max()
        plt.plot(tmp, lw=3, c='k')
        x = minimize(lambda x: np.sum((tmp - x[0] - x[1] * a[k, :770])**2),
                     [1.1 * (2 - i), 1 / a[k, :770].max()], method='Nelder-Mead')['x']
        plt.plot(x[0] + x[1] * a[k], lw=1.5, c=col[2 * i])
    plt.xlim(0, 770)
    plt.ylim(0, 3.2)
    plt.xticks([0, 600], [0, 5])
    plt.yticks([])
    plt.xlabel('Time [min]', labelpad=-15)
    plt.ylabel('Fluorescence [a.u.]')
    simpleaxis(plt.gca())
    plt.subplots_adjust(.03, .13, .988, .99)
    if figpath:
        plt.savefig(figpath + '/seriesNMF.pdf' if j else figpath + '/seriesQNMF.pdf')
    else:
        showpause()


# plot shapes for short and long processing time
fig = plt.figure(figsize=(10, 7.5))
for j, s in enumerate([shapes, shapesQ]):
    for i, k in enumerate(idx):
        ax = fig.add_axes([[0, .375, .64][i], .5 * j, .36, .5])
        ax.imshow(s[k][map(lambda a: slice(*a), boxes[k])].T,
                  cmap='hot', interpolation='nearest')
        ax.scatter([1.5], [1.5], s=500, marker='x',
                   lw=10, c=col[2 * i], zorder=[-11, 11][j])
        ax.axis('off')
        plt.xlim(-.5, 18.5)
plt.savefig(figpath + '/shapes.pdf') if figpath else plt.show(block=True)
