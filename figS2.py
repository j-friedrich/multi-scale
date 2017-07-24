import numpy as np
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tifffile
import psutil
from operator import itemgetter
from scipy.sparse import coo_matrix, csc_matrix
from scipy.ndimage.filters import uniform_filter
from skimage.transform import downscale_local_mean
from CNMF import LocalNMF, HALS4activity
from functions import init_fig, simpleaxis, showpause, IQRfill
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper
from scipy.stats import pearsonr


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


def hals(Y, A, b, bSiz=3, maxIter=10):
    d1, d2, T = np.shape(Y)
    f = np.ones((1, T))
    C = np.ones((A.shape[1], T))
    ind_A = uniform_filter(np.reshape(
        A, (d1, d2, A.shape[1]), order='F'), size=(bSiz, bSiz, 0))
    ind_A = np.reshape(ind_A > 1e-10, (d1 * d2, A.shape[1]))
    ind_A = csc_matrix(ind_A)  # indicator of nonnero pixels
    K = np.shape(A)[1]  # number of neurons

    def HALS4activity(data, S, activity):
        A = S.dot(data)
        B = S.dot(S.T).toarray()
        for _ in range(maxIter):
            for mcell in range(K + 1):  # neurons and background
                activity[mcell] += (A[mcell] - np.dot(B[mcell].T, activity)) / B[mcell, mcell]
                activity[mcell][activity[mcell] < 0] = 0
        return activity
    Ab = coo_matrix(np.c_[A, b].T)
    Cf = np.r_[C, f.reshape(1, -1)]
    Cf = HALS4activity(np.reshape(Y, (d1 * d2, T)), Ab, Cf)
    return Cf[:-1], Cf[-1].reshape(1, -1)


# ########################################################
# ####################  light-sheet  #####################
# ########################################################

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
trueC = activity.copy()
trueA = shapes.copy()


# Poisson
signal = trueC.T.dot(trueA.reshape(N + 1, -1)).astype('float32').reshape(data.shape)
np.random.seed(0)
YrPoisson = np.random.poisson(signal - 55.66).astype('float32')
YrPoisson -= YrPoisson.min()

# get shapes on data with high spatial resolution
MSE_array, shapesPoisson, activity, boxes = LocalNMF(
    YrPoisson, centers, sig, iters=5, mb=20, iters0=30, ds=[3, 3])
# scale
z = np.sqrt(np.sum(shapes.reshape(len(shapes), -1)[:-1]**2, 1)).reshape(-1, 1)
activity[:-1] *= z
shapesPoisson[:-1] /= z.reshape(-1, 1, 1)

# decimate
activityPoisson = {}
dsls = [1, 2, 3, 4, 6, 8, 12]
for ds in dsls:
    activityPoisson[ds] = HALS4activity(YrPoisson.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                                        .mean(-1).mean(-2).reshape(len(YrPoisson), -1),
                                        shapesPoisson.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                                        .mean(-1).mean(-2).reshape(len(shapesPoisson), -1),
                                        activity.copy(), 10)


# Gauss
signal = trueC.T.dot(trueA.reshape(N + 1, -1)).astype('float32').reshape(data.shape)
np.random.seed(0)
YrGauss = signal + np.random.randn(*data.shape).astype('float32') * \
    np.std(data - signal).astype('float32')
YrGauss -= YrGauss.min()
print(np.var(data - signal), np.var(YrPoisson - signal), np.var(YrGauss - signal))

# get shapes on data with high spatial resolution
MSE_array, shapesGauss, activity, boxes = LocalNMF(
    YrGauss, centers, sig, iters=5, mb=20, iters0=30, ds=[3, 3])
# scale
z = np.sqrt(np.sum(shapes.reshape(len(shapes), -1)[:-1]**2, 1)).reshape(-1, 1)
activity[:-1] *= z
shapesGauss[:-1] /= z.reshape(-1, 1, 1)

# decimate
activityGauss = {}
dsls = [1, 2, 3, 4, 6, 8, 12]
for ds in dsls:
    activityGauss[ds] = HALS4activity(YrGauss.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                                      .mean(-1).mean(-2).reshape(len(YrGauss), -1),
                                      shapesGauss.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                                      .mean(-1).mean(-2).reshape(len(shapesGauss), -1),
                                      activity.copy(), 10)


# plot correlation
corPoisson = np.array([[np.corrcoef(trueC[i], activityPoisson[ds][i])[0, 1]
                        for i in range(N)] for ds in dsls])
corGauss = np.array([[np.corrcoef(trueC[i], activityGauss[ds][i])[0, 1]
                      for i in range(N)] for ds in dsls])

for i, cor in enumerate([corPoisson, corGauss]):
    plt.plot(dsls, np.median(cor, 1), lw=4, c=[orange, cyan][i], label=['Poisson', 'Gaussian'][i])
    IQRfill(cor.T, dsls, [orange, cyan][i])
plt.xlabel('Spatial decimation')
ax = plt.gca()
simpleaxis(ax)
ax.patch.set_visible(False)
plt.xticks(dsls, ['1', '', '', '4x4', '6x6', '8x8', '12x12'])
plt.yticks([.95, 1.00], [.95, '1'])
plt.xlim(dsls[0], dsls[-1])
plt.ylim(.949, 1)
plt.ylabel('Correlation w/ ground truth', y=.54, labelpad=-25)
plt.legend(loc=(.01, .25), ncol=1)
plt.subplots_adjust(.105, .15, .94, .97)
plt.savefig(figpath + '/Poisson-Gauss_zebrafish.pdf') if figpath else showpause()


#

# ########################################################
# #####################  two-photon  #####################
# ########################################################

# Load results

cluster = False  # whether code is run on cluster
dsls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

f, A2, b2, C2 = itemgetter('f', 'A2', 'b2', 'C2')(np.load('results/CNMF-HRshapes.npz'))
A2 = A2.item()
N = A2.shape[1]

res = np.load('results/decimate.npz')
ssub = res['ssub'].item()
srt = cse.utilities.order_components(A2, ssub[1][0])[-1]
trueC = ssub[1][0]
trueSpikes = ssub[1][2]


# artificial data, from 'true' generative model by drawing from Poisson or Gaussian
# load results if saved, otherwise generate them

for noise in ('gauss', 'poisson'):
    try:
        res = np.load('results/decimate-%s.npz' % noise)
        if noise == 'gauss':
            ssub, ssubX = res['ssub'].item(), res['ssubX'].item()
        else:
            ssub0, ssubX0 = res['ssub'].item(), res['ssubX'].item()
    except:
        n_processes = np.maximum(psutil.cpu_count() - 2, 1)
        print "Stopping  cluster to avoid unnencessary use of memory...."
        cse.utilities.stop_server()

        Yr = tifffile.TiffFile('180um_20fps_350umX350um.tif').asarray().astype(dtype=np.float32)
        Yr = np.transpose(Yr, (1, 2, 0))
        d1, d2, T = Yr.shape
        Yr = np.reshape(Yr, (d1 * d2, T), order='F')
        np.save('Yr', Yr)
        Yr = np.load('Yr.npy', mmap_mode='r')
        Y = np.reshape(Yr, (d1, d2, T), order='F')

        options = cse.utilities.CNMFSetParms(Y, n_processes, gSig=[6, 6], K=50, tsub=30, ssub=2)

        if cluster:
            cse.utilities.start_server()

        # ## infer shapes on full, half or quarter of original high-res data
        # ## load result
        A2, b2, C2, f, A_m, C_m, sn = itemgetter(
            'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(np.load('results/CNMF-HRshapes.npz'))
        A2 = A2.item()
        N = A2.shape[1]
        options['temporal_params']['ITER'] = 5

        # ## load or generate noisy data
        try:
            Yr = np.load('Yr%s.npy' % noise, mmap_mode='r')
        except:
            Yr = np.load('Yr.npy', mmap_mode='r')
            # load shapes inferred on original high-res data
            A2, b2, C2, f, A_m, C_m, sn = itemgetter(
                'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(np.load('results/CNMF-HRshapes.npz'))
            A2 = A2.item()
            ssub1 = np.load('results/decimate.npz')['ssub'].item()[1]
            # ssub1 is tuple (C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA)
            signal = A2.dot(ssub1[0]).astype('float32') + b2.dot(ssub1[1]).astype('float32')
            if noise == 'gauss':
                np.random.seed(0)
                YrGauss = signal + np.random.randn(*Yr.shape) * (Yr - signal).std()
                np.save('Yr%s' % noise, YrGauss)
                del YrGauss
            else:
                np.random.seed(0)
                YrPoisson = np.random.poisson(np.var(Yr - signal) + signal).astype('float32')
                YrPoisson -= YrPoisson.min()
                np.save('Yr%s' % noise, YrPoisson)
                del YrPoisson
            del signal
            Yr = np.load('Yr%s.npy' % noise, mmap_mode='r')
        # ## get spatial components on high-resolution data
        try:  # load shapes inferred on shuffled high-res data
            A2, b2, C2, f, A_m, C_m, sn = itemgetter(
                'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(
                np.load('results/CNMF-HRshapes-%s.npz' % noise))
            A2 = A2.item()
        except:
            A2, b2, C2, f, A_m, C_m, sn = itemgetter(
                'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(np.load('results/CNMF-HRshapes.npz'))

            # refine spatial components
            t1 = time()
            A2, b2, C2 = cse.spatial.update_spatial_components(
                Yr, C_m, f, np.ravel(A_m)[0], sn=sn, **options['spatial_params'])
            print time() - t1
            print('DONE!')

            # normalize
            z = np.linalg.norm(A2.toarray(), 2, 0)
            A2 = coo_matrix(A2.toarray() / z)
            C2 *= z.reshape(-1, 1)

            # save intermediate results
            np.savez_compressed('results/CNMF-HRshapes-%s.npz' % noise,
                                **{'A2': A2, 'b2': b2, 'C2': C2, 'f': f,
                                   'A_m': A_m, 'C_m': C_m, 'sn': sn})

        # infer temporal components for various downsampling scenarios
        ssub = {}
        ssubX = {}
        t1 = time()
        for ds in dsls:
            C0, f0 = hals(downscale_local_mean(Yr.reshape(d1, d2, T), (ds, ds, 1)),
                          downscale_local_mean(A2.toarray().reshape(
                              d1, d2, N), (ds, ds, 1)).reshape(-1, N),
                          downscale_local_mean(b2.reshape(d1, d2, 1), (ds, ds, 1)).reshape(-1, 1))
            ssub[ds] = cse.temporal.update_temporal_components(
                downscale_local_mean(Yr.reshape(d1, d2, T), (ds, ds, 1)).reshape(-1, T),
                downscale_local_mean(A2.toarray().reshape(d1, d2, N), (ds, ds, 1)).reshape(-1, N),
                downscale_local_mean(b2.reshape(d1, d2, 1), (ds, ds, 1)).reshape(-1, 1),
                C0, f0, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
            C0, f0 = hals(downscale_local_mean(Yr.reshape(d1, d2, T), (ds, 1, 1)),
                          downscale_local_mean(A2.toarray().reshape(
                              d1, d2, N), (ds, 1, 1)).reshape(-1, N),
                          downscale_local_mean(b2.reshape(d1, d2, 1), (ds, 1, 1)).reshape(-1, 1))
            ssubX[ds] = ssub[ds] if ds == 1 else cse.temporal.update_temporal_components(
                downscale_local_mean(Yr.reshape(d1, d2, T), (ds, 1, 1)).reshape(-1, T),
                downscale_local_mean(A2.toarray().reshape(d1, d2, N), (ds, 1, 1)).reshape(-1, N),
                downscale_local_mean(b2.reshape(d1, d2, 1), (ds, 1, 1)).reshape(-1, 1),
                C0, f0, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
            print time() - t1
        print('DONE!')

        # save results
        np.savez_compressed('results/decimate-%s.npz' % noise, **{'ssub': ssub, 'ssubX': ssubX})
        # load results
        res = np.load('results/decimate-%s.npz' % noise)
        if noise == 'gauss':
            ssub, ssubX = res['ssub'].item(), res['ssubX'].item()
        else:
            ssub0, ssubX0 = res['ssub'].item(), res['ssubX'].item()

#

# # plot correlations

r = pearsonr
plt.figure()


def foo(ssub, comp, dsls=dsls, ca_or_spikes='ca'):
    N, T = comp.shape
    cor = np.zeros((N, len(dsls))) * np.nan
    for i, ds in enumerate(dsls):
        if len(ssub[ds][0]) == len(comp):
            cor[:, i] = np.array(
                [r(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                   comp[n])[0] for n in range(N)])
    return np.nan_to_num(cor)


corP = foo(ssub0, trueC, dsls)
l1, = plt.plot(dsls, np.median(corP, 0), lw=4, c=orange, label='Poisson')
IQRfill(corP, dsls, orange)

corG = foo(ssub, trueC)
l2, = plt.plot(dsls, np.median(corG, 0), lw=4, c=cyan, label='Gaussian')
IQRfill(corG, dsls, cyan)

corPs = foo(ssub0, trueSpikes, dsls, ca_or_spikes='spikes')
plt.plot(dsls, np.median(corPs, 0), lw=4, c=orange, ls='--')
IQRfill(corPs, dsls, orange, ls='--', hatch='\\\\\\')

corGs = foo(ssub, trueSpikes, ca_or_spikes='spikes')
plt.plot(dsls, np.median(corGs, 0), lw=4, c=cyan, ls='--')
IQRfill(corGs, dsls, cyan, ls='--', hatch='///')

l3, = plt.plot([0, 1], [-1, -1], lw=4, c='k', label='denoised')
l4, = plt.plot([0, 1], [-1, -1], lw=4, c='k', ls='--', label='deconvolved')

plt.xlabel('Spatial decimation')
simpleaxis(plt.gca())
plt.xticks(dsls, ['1', '', '', '', '', '8x8', '', '16x16', '24x24', '32x32'])
plt.yticks(*[np.round(np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.2), 1)] * 2)
plt.xlim(dsls[0], dsls[-1])
plt.ylim(.4, 1)
plt.ylabel(r'Correlation w/ ground truth', labelpad=1)
plt.legend(handles=[l3, l4, l1, l2], loc=(.01, .01), ncol=1)
plt.subplots_adjust(.125, .15, .94, .97)
plt.savefig(figpath + '/Poisson-Gauss.pdf') if figpath else showpause()

plt.figure()
plt.scatter(corP[:, -3], corG[:, -3], marker='x', c=vermillon, label='denoised')
plt.scatter(corPs[:, -3], corGs[:, -3], marker='o', c=blue, label='deconvolved')
plt.legend(loc=(.01, .8))
simpleaxis(plt.gca())
plt.xticks(*[[0, .5, 1.0]] * 2)
plt.yticks(*[[0, .5, 1.0]] * 2)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Correlation for Poisson noise')
plt.ylabel('Correlation for Gaussian noise', labelpad=4)
plt.subplots_adjust(.13, .15, .97, .97)
plt.savefig(figpath + '/Poisson-Gauss_scatter.pdf') if figpath else plt.show(block=True)
