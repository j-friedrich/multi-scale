import numpy as np
from time import time
from scipy.sparse import coo_matrix, csc_matrix
from scipy.ndimage.filters import uniform_filter
import tifffile
import psutil
from operator import itemgetter
from skimage.transform import downscale_local_mean
import ca_source_extraction as cse  # https://github.com/j-friedrich/CaImAn/tree/multi-scale_paper
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
from functions import init_fig, simpleaxis, showpause, IQRfill

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


cluster = False  # whether code is run on cluster
dsls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]  # downsample factors

n_processes = np.maximum(psutil.cpu_count() - 2, 1)
print "Stopping  cluster to avoid unnencessary use of memory...."
cse.utilities.stop_server()

Yr = tifffile.TiffFile('180um_20fps_350umX350um.tif').asarray().astype(dtype=np.float32)
Yr = np.transpose(Yr, (1, 2, 0))
d1, d2, T = Yr.shape
Yr = np.reshape(Yr, (d1 * d2, T), order='F')
Y = np.reshape(Yr, (d1, d2, T), order='F')

options = cse.utilities.CNMFSetParms(Y, n_processes, gSig=[6, 6], K=50, tsub=30, ssub=2)

if cluster:
    cse.utilities.start_server()


for gcamp in ('6f', '6s'):
    try:  # load saved chen results
        trueC, trueSpikes = itemgetter('C', 'S')(np.load('results/chen%s-truth.npz' % gcamp))
        ssub = np.load('results/decimate-chen%s.npz' % gcamp)['ssub'].item()
        ssub0 = np.load('results/decimate-chen%sLR.npz' % gcamp)['ssub'].item()
    except:  # generate data and run analysis
        # ## load 2P result
        A2, b2, C2, f, A_m, C_m, sn = itemgetter(
            'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(np.load(
                'results/CNMF-HRshapes.npz'))
        A2 = A2.item()
        N = A2.shape[1]
        options['temporal_params']['ITER'] = 5

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
                        activity[mcell] += (A[mcell] - np.dot(B[mcell].T,
                                                              activity)) / B[mcell, mcell]
                        activity[mcell][activity[mcell] < 0] = 0
                return activity
            Ab = coo_matrix(np.c_[A, b].T)
            Cf = np.r_[C, f.reshape(1, -1)]
            Cf = HALS4activity(np.reshape(Y, (d1 * d2, T)), Ab, Cf)
            return Cf[:-1], Cf[-1].reshape(1, -1)

        # ## load or generate shuffled data
        try:
            Yr = np.load('YrChen%s.npy' % gcamp, mmap_mode='r')
            C, S = itemgetter('C', 'S')(np.load('results/chen%s-truth.npz' % gcamp))
        except:
            # load shapes inferred on original high-res data
            A2, b2, C2, f, A_m, C_m, sn = itemgetter(
                'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(
                np.load('results/CNMF-HRshapes.npz'))
            A2 = A2.item()
            ssub1 = np.load('results/decimate.npz')['ssub'].item()[1]
            # ssub1 is tuple (C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA)
            if gcamp == '6f':
                dataset = 7
                tau = 400
            else:
                dataset = 8
                tau = 600
            b = True
            spikes_train = pd.read_csv('../../spikefinder-python/spikefinder.train/' +
                                       str(dataset) + '.train.spikes.csv')
            calcium_train = pd.read_csv('../../spikefinder-python/spikefinder.train/' +
                                        str(dataset) + '.train.calcium.csv')
            trueS = np.array(spikes_train).T
            Y = np.array(calcium_train).T
            k = []
            for n in range(len(trueS)):
                s = trueS[n]
                s = s[~np.isnan(s)]
                t = len(s)
                ss = np.zeros((tau, t))
                for i in range(tau):
                    ss[i, i:] = s[:t - i]
                ssm = ss - ss.mean() if b else ss
                k.append(np.linalg.inv(ssm.dot(ssm.T)).dot(
                    ssm.dot(Y[n][:t]))[:(300 if gcamp == '6f' else 500)])
            if gcamp == '6f':
                k = np.reshape(k, (-1, 60, 5)).mean(-1)
                k -= k.min(1)[:, None]
            else:
                k = np.reshape(k, (-1, 100, 5)).mean(-1)
            signal = A2.dot(ssub1[0]).astype('float32') + b2.dot(ssub1[1]).astype('float32')
            darkcounts = np.var(Yr - signal)
            trueS = trueS[:, :trueS.shape[1] // 5 * 5].reshape(len(trueS), -1, 5).sum(-1)
            C = np.zeros((N, T), dtype='float32')
            S = np.zeros((N, T), dtype='uint8')
            np.random.seed(0)
            for n in range(N):
                tmp = trueS[n % len(trueS)]
                tmp = tmp[~np.isnan(tmp)]
                i = np.random.randint(len(tmp) - T)
                while tmp[i:i + T].sum() < 3:
                    i = np.random.randint(len(tmp) - T)
                S[n] = tmp[i:i + T]
                C[n] = np.convolve(S[n], k[n % len(trueS)], 'full')[:T]
                C[n] = (C[n] / C[n].max() * ssub1[0][n].max()).astype('float32')
            signal = A2.dot(C).astype('float32') + b2.dot(ssub1[1]).astype('float32')
            darkcounts = 600000
            YrChen = np.random.poisson(darkcounts + signal).astype('float32')
            YrChen -= YrChen.min()
            np.save('YrChen%s' % gcamp, YrChen)
            del YrChen
            del signal
            Yr = np.load('YrChen%s.npy' % gcamp, mmap_mode='r')
            np.savez_compressed('results/chen%s-truth.npz' % gcamp, **{'C': C, 'S': S})

        # ## get spatial components on high-resolution data
        try:  # load shapes inferred on high-res data
            A2, b2, sn = itemgetter('A2', 'b2', 'sn')(
                np.load('results/CNMF-HRshapes-chen%s.npz' % gcamp))
            A2 = A2.item()
        except:
            f, A_m = itemgetter('f', 'A_m')(np.load('results/CNMF-HRshapes.npz'))
            sn = cse.pre_processing.preprocess_data(Yr, **options['preprocess_params'])[1]

            # refine spatial components
            t1 = time()
            A2, b2, C2 = cse.spatial.update_spatial_components(
                Yr, C, f, np.ravel(A_m)[0], sn=sn, **options['spatial_params'])
            print time() - t1
            print('DONE!')

            # normalize
            z = np.linalg.norm(A2.toarray(), 2, 0)
            A2 = coo_matrix(A2.toarray() / z)
            C2 *= z.reshape(-1, 1)

            # save intermediate results
            np.savez_compressed('results/CNMF-HRshapes-chen%s.npz' % gcamp,
                                **{'A2': A2, 'b2': b2, 'sn': sn})

        # infer temporal components for various downsampling scenarios
        ssub = {}
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
            print time() - t1
        print('DONE!')

        # save results
        np.savez_compressed('results/decimate-chen%s.npz' % gcamp, **{'ssub': ssub})

        # recalculate sn for shape update on decimated data
        # decimate along x and y
        f, A_m = itemgetter('f', 'A_m')(np.load('results/CNMF-HRshapes.npz'))
        t1 = time()
        A2 = {}
        b2 = {}
        C2 = {}
        n_pixels_per_process = options['spatial_params']['n_pixels_per_process']
        for ds in dsls:
            tmp = downscale_local_mean(Yr.reshape(d1, d2, T), (ds, ds, 1))
            options['spatial_params']['dims'] = tmp.shape[:2]
            options['spatial_params']['n_pixels_per_process'] = n_pixels_per_process / ds**2
            options['preprocess_params']['n_pixels_per_process'] = n_pixels_per_process / ds**2
            A2[ds], b2[ds], C2[ds] = cse.spatial.update_spatial_components(
                tmp.reshape(-1, T), C, f,
                downscale_local_mean(np.ravel(A_m)[0].toarray().
                                     reshape(d1, d2, N), (ds, ds, 1)).reshape(-1, N),
                sn=cse.pre_processing.preprocess_data(Yr, **options['preprocess_params'])[1],
                **options['spatial_params'])
            print time() - t1
        print('DONE!')
        options['spatial_params']['dims'] = d1, d2
        options['spatial_params']['n_pixels_per_process'] = n_pixels_per_process
        options['preprocess_params']['n_pixels_per_process'] = n_pixels_per_process

        ssub = {}
        t1 = time()
        for ds in dsls:
            C0, f0 = hals(downscale_local_mean(Yr.reshape(d1, d2, T),
                                               (ds, ds, 1)), A2[ds].toarray(), b2[ds])
            ssub[ds] = cse.temporal.update_temporal_components(
                downscale_local_mean(Yr.reshape(d1, d2, T), (ds, ds, 1)).reshape(-1, T),
                A2[ds], b2[ds], C0, f0, bl=None, c1=None, sn=None, g=None,
                **options['temporal_params'])
            print time() - t1
        print('DONE!')

        # save results
        np.savez_compressed('results/decimate-chen%sLR.npz' % gcamp, **{'ssub': ssub})

        trueC, trueSpikes = itemgetter('C', 'S')(np.load('results/chen%s-truth.npz' % gcamp))
        ssub = np.load('results/decimate-chen%s.npz' % gcamp)['ssub'].item()
        ssub0 = np.load('results/decimate-chen%sLR.npz' % gcamp)['ssub'].item()

    # plot correlations

    def plotCorr(ssub, ssub0, r=pearsonr, ds1phase=[1, 2, 3, 4, 6, 8], loc=(.1, .01)):
        def foo(ssub, comp, dsls=dsls, ca_or_spikes='ca'):
            N, T = comp.shape
            cor = np.zeros((N, len(dsls))) * np.nan
            for i, ds in enumerate(dsls):
                if len(ssub[ds][0]) == len(comp):
                    cor[:, i] = np.array(
                        [r(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                           comp[n])[0] for n in range(N)])
                else:  # necessary if update_spatial_components removed a component
                    mapIdx = [np.argmax([np.corrcoef(s, tC)[0, 1] for tC in comp])
                              for s in ssub[ds][0 if ca_or_spikes == 'ca' else 2]]
                    for n in range(len(ssub[ds][0])):
                        cor[mapIdx[n], i] = np.array(
                            r(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                              comp[mapIdx[n]])[0])
            return np.nan_to_num(cor)

        cor = foo(ssub0, trueC, ds1phase)
        l1, = plt.plot(ds1phase, np.median(np.nan_to_num(cor), 0), lw=4, c=cyan,
                       label='1 phase imaging')
        IQRfill(cor, dsls, cyan)

        cor = foo(ssub, trueC)
        l2, = plt.plot(dsls, np.median(cor, 0), lw=4, c=orange, label='2 phase imaging')
        IQRfill(cor, dsls, orange)

        cor = foo(ssub0, trueSpikes, ds1phase, ca_or_spikes='spikes')
        plt.plot(ds1phase, np.median(cor, 0), lw=4, c=cyan, ls='--')
        IQRfill(cor, dsls, cyan, ls='--', hatch='///')

        cor = foo(ssub, trueSpikes, ca_or_spikes='spikes')
        plt.plot(dsls, np.median(cor, 0), lw=4, c=orange, ls='--')
        IQRfill(cor, dsls, orange, ls='--', hatch='\\\\\\')

        l3, = plt.plot([0, 1], [-1, -1], lw=4, c='k', label='denoised')
        l4, = plt.plot([0, 1], [-1, -1], lw=4, c='k', ls='--', label='deconvolved')

        plt.xlabel('Spatial decimation')
        simpleaxis(plt.gca())
        plt.xticks(dsls, ['1', '', '', '', '', '8x8', '', '16x16', '24x24', '32x32'])
        plt.ylim(.4 if gcamp == '6f' else .2, 1)
        plt.yticks(*[np.round(np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.2), 1)] * 2)
        plt.xlim(dsls[0], dsls[-1])
        plt.ylabel('Correlation w/ ground truth', y=.5, labelpad=4)
        plt.legend(handles=[l3, l4, l1, l2], loc=loc, ncol=1)
        plt.subplots_adjust(.13, .15, .94, .96)
        return l1, l2, l3, l4

    plt.figure()
    plotCorr(ssub, ssub0, loc=(.2, .01))
    plt.savefig(figpath + '/Chen%s.pdf' % gcamp) if figpath else showpause()


# STOP CLUSTER
cse.utilities.stop_server()
