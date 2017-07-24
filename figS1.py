import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from skimage.filters import gaussian

try:
    from sys import argv
    from os.path import isdir
    figpath = argv[1] if isdir(argv[1]) else False
except:
    figpath = False

d1, d2 = (512, 512)


# Load shapes

def GetBox(centers, R, dims):
    D = len(R)
    box = np.zeros((D, 2), dtype=int)
    for dd in range(D):
        box[dd, 0] = max((centers[dd] - R[dd], 0))
        box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
    return box


A2 = np.load('results/CNMF-HRshapes.npz')['A2'].item()
shapes = A2.T.toarray().reshape(-1, d1, d2)
boxes = [GetBox(center_of_mass(s), [17, 17], [d1, d2]) for s in shapes]


# load series and calc corr

def foo(ssub, comp, ds=16):
    N, T = comp.shape
    cor = np.zeros(N) * np.nan
    if len(ssub[ds][0]) == len(comp):
        cor = np.array([np.corrcoef(ssub[ds][0][n], comp[n])[0, 1] for n in range(N)])
    else:  # necessary if update_spatial_components removed a component
        mapIdx = [np.argmax([np.corrcoef(s, tC)[0, 1] for tC in comp]) for s in ssub[ds][0]]
        for n in range(len(ssub[ds][0])):
            cor[mapIdx[n]] = np.array(
                np.corrcoef(ssub[ds][0][n], comp[mapIdx[n]])[0, 1])
    return np.nan_to_num(cor)


ssub = np.load('results/decimate.npz')['ssub'].item()
cor = foo(ssub, ssub[1][0])


# plot

srt = np.argsort(cor)
plt.figure(figsize=(17, 11))
for i, s in enumerate(srt[::-1]):
    plt.subplot(11, 17, 1 + i)
    plt.imshow(gaussian(shapes[s], 1)[map(lambda a: slice(*a), boxes[s])],
               cmap='hot', interpolation='nearest')
    plt.axis('off')
plt.subplots_adjust(0, 0, 1, 1, .02, .02)
plt.savefig(figpath + '/shapes_sorted_by_corr16x16.pdf') if figpath else plt.show(block=True)
