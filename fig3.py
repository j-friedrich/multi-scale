import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CNMF import LocalNMF, HALS4activity
from functions import init_fig, simpleaxis, IQRfill
import ca_source_extraction as cse  # https://github.com/j-friedrich/CaImAn/tree/multi-scale_paper
import itertools

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
        cor[:, i] = np.array([np.corrcoef(ssub[ds][int(mapIdx[n])], comp[n])[0, 1]
                              for n in range(N)])
        mapIdxs[:, i] = mapIdx
    return cor, mapIdxs


fig = plt.figure(figsize=(6.5, 6))
for ssub, col in ((activityDSlr, cyan), (activityDS, orange)):
    cor, mx = foo(ssub, activityDS[1], shapes)
    if col == cyan:
        mapIdx = mx
    plt.plot(dsls, np.median(cor, 0), lw=3, c=col, clip_on=False,
             label='1 phase\n imaging' if col == cyan else '2 phase\n imaging')
    IQRfill(cor, dsls, col)
plt.xlabel('Spatial decimation')
ax = plt.gca()
simpleaxis(ax)
ax.patch.set_visible(False)
plt.xticks(dsls, ['1x1', '', '', '4x4', '6x6', '8x8', '12x12'])
plt.ylim(.55, 1)
plt.yticks(*[[.6, .8, 1.]] * 2)
plt.ylabel('Correlation', labelpad=0)
plt.legend(loc=(.01, .25), ncol=1)
plt.subplots_adjust(.145, .15, .885, .97)
ax2 = ax.twinx()
ax2.set_zorder(-1)
ax2.spines['top'].set_visible(False)
z = np.mean((activityDS[1].T.dot(shapes.reshape(len(shapes), -1)) -
             data.reshape(activity.shape[1], -1))**2)
ax2.plot(dsls, [np.mean((activityDSlr[ds].T.dot(shapesDSlr[ds].repeat(ds, 1).repeat(ds, 2).reshape(
    len(shapes), -1)) - data.reshape(activity.shape[1], -1))**2) / z for ds in dsls],
    '+--', lw=3, c=cyan, mew=3, ms=10, clip_on=False, zorder=5)
ax2.plot(dsls, [np.mean((activityDS[ds].T.dot(shapes.reshape(len(shapes), -1)) -
                         data.reshape(activity.shape[1], -1))**2) / z for ds in dsls],
         'x--', lw=3, c=orange, mew=3, ms=10, clip_on=False, zorder=5)
plt.ylim(.999, 1.03)
plt.xlim(dsls[0] - .2, dsls[-1] + .3)
plt.yticks([1.00, 1.03], ['1.00', '1.03'])
plt.ylabel('Normalized MSE', labelpad=-33)
plt.savefig(figpath + '/Corr_zebrafish.pdf') if figpath else plt.show(block=True)
