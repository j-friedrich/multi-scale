import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CNMF import LocalNMF
from functions import init_fig, simpleaxis
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper

init_fig()
save_figs = True #False  # subfolder fig must exist if set to True

# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
# vermillon, orange, yellow, green, cyan, blue, purple, grey

cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
gfp = matplotlib.colors.LinearSegmentedColormap('GFP_colormap', cdict, 256)


# load data
data = np.load('data_zebrafish.npy')
sig = (3, 3)
# find neurons greedily
N = 46
centers = cse.greedyROI(data.reshape((-1, 30) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[4, 4])[2]


# run NMF
MSE_array, shapes, activity, boxes = LocalNMF(data, centers, sig, verbose=True,
                                              iters=100, iters0=30, mb=30, ds=[3, 3])
MSE_array0, shapes0, activity0, boxes0 = LocalNMF(data, centers, sig, verbose=True,
                                                  iters=100, iters0=0, mb=1)
MSE_arrayQ, shapesQ, activityQ, boxesQ = LocalNMF(data, centers, sig, verbose=True,
                                                  iters=5, iters0=30, mb=30, ds=[3, 3])


# plot patch
idx = [1, 5, 27]
np.random.seed(0)
fig = plt.figure(figsize=(3, 3), frameon=False)
plt.imshow(np.sqrt(data.reshape(
    (-1, 10) + data.shape[1:]).mean(1).max(0)).T, gfp)
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
if save_figs:
    plt.savefig('fig/patch.pdf')
plt.show()


# # plot series for long processing time
# plt.figure(figsize=(18, 6. / 8 * 7))
# for i, k in enumerate(idx):
#     plt.plot(1.1 * i + activity0[k] /
#              activity0[k, 1200:2400].max(), lw=3, c='k')
#     plt.plot(1.1 * i + activity[k] / activity[k, 1200:2400].max(), lw=1.5, c=col[2 * i])
# plt.xlim(1200, 2400)
# plt.ylim(0, 3.2)
# plt.xticks([1200, 2400], [10, 20])
# plt.yticks([])
# plt.xlabel('Time [min]', labelpad=-15)
# plt.ylabel('Fluorescence [a.u.]')
# simpleaxis(plt.gca())
# plt.subplots_adjust(.03, .13, .988, .99)
# if save_figs:
#     plt.savefig('fig/seriesNMF.pdf')
# plt.show()


# plot series for short and long processing time
for j, a in enumerate([activityQ, activity]):
    plt.figure(figsize=(18, 6. / 8 * 7))
    for i, k in enumerate(idx):
        plt.plot(1.1 * i + activity0[k] /
                 activity0[k, 1200:2400].max(), lw=3, c='k')
        plt.plot(1.1 * i + activity[k] / activity[k, 1200:2400].max(), lw=1.5, c=col[2 * i])
    plt.xlim(1200, 2400)
    plt.ylim(0, 3.2)
    plt.xticks([1200, 2400], [10, 20])
    plt.yticks([])
    plt.xlabel('Time [min]', labelpad=-15)
    plt.ylabel('Fluorescence [a.u.]')
    simpleaxis(plt.gca())
    plt.subplots_adjust(.03, .13, .988, .99)
    if save_figs:
        plt.savefig('fig/seriesQNMF.pdf' if j == 0 else 'fig/seriesNMF.pdf')
    plt.show()


# # plot shapes for long processing time
# fig = plt.figure(figsize=(9, 6))
# for i, k in enumerate(idx):
#     ax = fig.add_axes([i / 3., .5, 1 / 3., .5])
#     ax.imshow(shapes[k][map(lambda a: slice(*a), boxes[k])].T,
#               cmap='hot', interpolation='nearest')
#     ax.scatter([16.5], [1.5], s=500, marker='x', lw=10, c=col[2 * i])
#     ax.axis('off')
#     ax = fig.add_axes([i / 3., 0, 1 / 3., .5])
#     ax.imshow(shapes0[k][map(lambda a: slice(*a), boxes0[k])].T,
#               cmap='hot', interpolation='nearest')
#     ax.scatter([16.5], [1.5], s=500, marker='x',
#                lw=10, c=col[2 * i], zorder=-11)
#     ax.axis('off')
# if save_figs:
#     plt.savefig('fig/shapesNMF.pdf')
# plt.show()


# # plot series after just 1 s processing time
# plt.figure(figsize=(18, 6. / 8 * 7))
# for i, k in enumerate(idx):
#     plt.plot(1.1 * i + activity0[k] /
#              activity0[k, 1200:2400].max(), lw=3, c='k')
#     plt.plot(1.1 * i + activityQ[k] / activityQ[k, 1200:2400].max(), lw=1.5, c=col[2 * i])
# plt.xlim(1200, 2400)
# plt.ylim(0, 3.2)
# plt.xticks([1200, 2400], [10, 20])
# plt.yticks([])
# plt.xlabel('Time [min]', labelpad=-15)
# plt.ylabel('Fluorescence [a.u.]')
# simpleaxis(plt.gca())
# plt.subplots_adjust(.03, .13, .988, .99)
# if save_figs:
#     plt.savefig('fig/seriesQNMF.pdf')
# plt.show()


# plot shapes for short and long processing time
for j, s in enumerate([shapesQ, shapes]):
    fig = plt.figure(figsize=(9, 6))
    for i, k in enumerate(idx):
        ax = fig.add_axes([i / 3., .5, 1 / 3., .5])
        ax.imshow(shapes[k][map(lambda a: slice(*a), boxes[k])].T,
                  cmap='hot', interpolation='nearest')
        ax.scatter([16.5], [1.5], s=500, marker='x', lw=10, c=col[2 * i])
        ax.axis('off')
        ax = fig.add_axes([i / 3., 0, 1 / 3., .5])
        ax.imshow(s[k][map(lambda a: slice(*a), boxes0[k])].T,
                  cmap='hot', interpolation='nearest')
        ax.scatter([16.5], [1.5], s=500, marker='x',
                   lw=10, c=col[2 * i], zorder=-11)
        ax.axis('off')
    if save_figs:
        plt.savefig('fig/shapesQNMF.pdf' if j == 0 else 'fig/shapesNMF.pdf')
    plt.show()

# # plot shapes after just 1 s processing time
# fig = plt.figure(figsize=(9, 6))
# for i, k in enumerate(idx):
#     ax = fig.add_axes([i / 3., .5, 1 / 3., .5])
#     ax.imshow(shapesQ[k][map(lambda a: slice(*a), boxes[k])].T,
#               cmap='hot', interpolation='nearest')
#     ax.scatter([16.5], [1.5], s=500, marker='x', lw=10, c=col[2 * i])
#     ax.axis('off')
#     ax = fig.add_axes([i / 3., 0, 1 / 3., .5])
#     ax.imshow(shapes0[k][map(lambda a: slice(*a), boxes0[k])].T,
#               cmap='hot', interpolation='nearest')
#     ax.scatter([16.5], [1.5], s=500, marker='x',
#                lw=10, c=col[2 * i], zorder=-11)
#     ax.axis('off')
# if save_figs:
#     plt.savefig('fig/shapesQNMF.pdf')
# plt.show()
