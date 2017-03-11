import numpy as np
import matplotlib.pyplot as plt
from CNMF import LocalNMF, HALS4activity
from functions import init_fig, simpleaxis, noclip, showpause
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper
import scipy
import itertools

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


# Fetch Data
data = np.load('data_zebrafish.npy')
sig = (3, 3)
# find neurons greedily
N = 49
A, C, b, f, centers = cse.initialization.initialize_components(data.transpose(1, 2, 0), N)


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


# get shapes not on high-res but coarse data
activityDSlr = {}
activityDSlr[1] = activityDS[1]
shapesDSlr = {}
shapesDSlr[1] = shapes
for ds in dsls[1:]:
    MSE_array, shapesLR, activity, _ = LocalNMF(
        data.reshape(-1, 96 / ds, ds, 96 / ds, ds).mean(-1).mean(-2),
        1. * centers / ds, 1. * np.array(sig) / ds, iters=5,
        mb=20, iters0=30, ds=[1 + 2 / ds, 1 + 2 / ds])
    activityDSlr[ds] = HALS4activity(
        data.reshape(-1, 96 / ds, ds, 96 / ds, ds)
        .mean(-1).mean(-2).reshape(len(data), -1),
        shapesLR.reshape(len(shapesLR), -1), activity.copy(), 10)
    shapesDSlr[ds] = shapesLR


# plot correlation values

def foo(ssub, comp, shapes):
    N, T = comp.shape
    cc = np.corrcoef(shapes.reshape(N, -1)[:-1]) > .2
    blocks = [set(np.where(c)[0]) for c in cc]
    for k in range(len(blocks)):
        for _ in range(10):
            for j in range(len(blocks) - 1, k, -1):
                if len(blocks[k].intersection(blocks[j])):
                    blocks[k] = blocks[k].union(blocks[j])
                    blocks.pop(j)
    blocks.append({N - 1})
    cor = np.zeros((N, len(dsls))) * np.nan
    mapIdxs = np.zeros((N, len(dsls)), dtype='uint8')
    for i, ds in enumerate(dsls):
        mapIdx = np.nan * np.zeros(N, dtype='uint8')
        corT = np.asarray([[np.corrcoef(s, tC)[0, 1]
                            for s in ssub[ds]] for tC in comp])
        q = None
        # first assign neurons that haven't switched
        for _ in range(10):
            if np.any(np.isnan(mapIdx)):
                nanIdx = np.where(np.isnan(mapIdx))[0]
                q = corT[np.isnan(mapIdx)][:, np.isnan(mapIdx)]
                for k in range(len(q)):
                    if (np.argmax(q[k]) == k and np.argmax(q[:, k]) == k)\
                            or (len(filter(lambda b: nanIdx[k] in b, blocks)[0]) == 1):
                        mapIdx[nanIdx[k]] = nanIdx[k]
        # check permutations of nearby neurons
        while np.any(np.isnan(mapIdx)):
            nanIdx = np.where(np.isnan(mapIdx))[0]
            block = filter(lambda b: nanIdx[0] in b, blocks)[0]
            idx = list(block.intersection(nanIdx))
            bestcorr = -np.inf
            for perm in itertools.permutations(idx):
                perm = list(perm)
                c = np.diag(corT[idx][:, perm]).sum()
                if c > bestcorr:
                    bestcorr = c
                    bestperm = perm
            mapIdx[list(idx)] = bestperm
        if np.any(mapIdx != range(N)):
            print ds, len(set(mapIdx)), mapIdx
        cor[:, i] = np.array([np.corrcoef(ssub[ds][mapIdx[n]], comp[n])[0, 1] for n in range(N)])
        mapIdxs[:, i] = mapIdx
    return cor, mapIdxs


fig = plt.figure(figsize=(6.5, 6))
cor, mapIdx = foo(activityDSlr, activityDS[1], shapes)
plt.plot(dsls, np.mean(cor, 0), lw=4, c=cyan,
         label='1 phase\n imaging', clip_on=False)
noclip(plt.errorbar(dsls, np.mean(cor, 0),
                    yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                    lw=3, capthick=2, fmt='o', c=cyan, clip_on=False))
cor, _ = foo(activityDS, activityDS[1], shapes)
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
plt.savefig(figpath + '/Corr_zebrafish.pdf') if figpath else showpause()


# plot traces

fig = plt.figure(figsize=(17, 6))
for k, neuronId in enumerate([0, 14, 36]):
    ax0 = fig.add_axes([.03, .71 - .3 * k, .967, .28])
    for i, ds in enumerate([1, 2, 4, 8]):
        l, = plt.plot(activityDS[ds][neuronId][:960], ['-', '-', '--', '-.'][i],
                      c=[green, yellow, vermillon, 'k'][i], label='%dx%d' % (ds, ds))
        if i == 1:
            l.set_dashes([14, 10])
    plt.legend(frameon=False, ncol=4, loc=[.05, .65], columnspacing=5.7, handletextpad=.1)
    z = scipy.optimize.minimize_scalar(
        lambda x: np.sum((activityDS[1][neuronId][:770] -
                          x * activityDSlr[8][mapIdx[neuronId, 5]][:770])**2))['x']
    plt.plot(z * activityDSlr[8][mapIdx[neuronId, 5]][:770], c=cyan, zorder=-11)
    plt.xticks(range(0, 900, 600), ['', '', ''])
    plt.xlim(0, 770)
    plt.ylim(0, plt.ylim()[1] * [1.25, 1.2, 1.1][k])
    tmp = [2, 7, 3][k]  # int(activityDS[ds][neuronId].max() / 100)-2
    # plt.yticks([0, 100 * tmp], [0, tmp])
    plt.yticks([])
    simpleaxis(plt.gca())
    for i, ds in enumerate([1, 2, 4, 8]):
        ax = fig.add_axes([.18 + i * .225, .82 - .3 * k, .04, .24])
        ss = (shapes[neuronId].reshape(shapes.shape[1] / ds, ds, shapes.shape[2] / ds, ds)
              .mean(-1).mean(-2).repeat(ds, 0).repeat(ds, 1))
        ax.imshow(ss[map(lambda a: slice(*a), boxes[neuronId])].T,
                  cmap='hot', interpolation='nearest')
        ax.axis('off')
        plt.xlim(0, 6 * sig[0])
        ax.text(1.2, .6, '%.3f' % (np.corrcoef(activityDS[ds][neuronId],
                                               activityDS[1][neuronId])[0, 1]),
                verticalalignment='center', transform=ax.transAxes)
ax0.set_xticklabels([0, 5])
ax0.set_xlabel('Time [min]', labelpad=-15)
ax0.set_ylabel('Fluorescence [a.u.]', y=1.55)
plt.savefig(figpath + '/traces_zebrafish.pdf') if figpath else plt.show(block=True)
