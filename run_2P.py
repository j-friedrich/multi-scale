import numpy as np
from time import time
from scipy.sparse import coo_matrix, csc_matrix
from scipy.ndimage.filters import uniform_filter
import tifffile
import psutil
from operator import itemgetter
from skimage.transform import downscale_local_mean
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper

cluster = False  # whether code is run on cluster
dsls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]  # downsample factors


# roughly number of cores on your machine minus 1
n_processes = np.maximum(psutil.cpu_count() - 2, 1)
print "Stopping  cluster to avoid unnencessary use of memory...."
cse.utilities.stop_server()

filename = '180um_20fps_350umX350um.tif'
t = tifffile.TiffFile(filename)
Yr = t.asarray().astype(dtype=np.float32)
Yr = np.transpose(Yr, (1, 2, 0))
d1, d2, T = Yr.shape
Yr = np.reshape(Yr, (d1 * d2, T), order='F')
np.save('Yr', Yr)
Yr = np.load('Yr.npy', mmap_mode='r')
Y = np.reshape(Yr, (d1, d2, T), order='F')

options = cse.utilities.CNMFSetParms(Y, n_processes, gSig=[6, 6], K=50, tsub=30, ssub=2)

if cluster:
    cse.utilities.start_server()

#

# ## infer shapes on full, half or quarter of original high-res data
for batch, batchname in enumerate(['', '-Aon1stHalf', '-Aon1stQuarter']):
    try:  # ## load result if saved
        A2, b2, C2, f, A_m, C_m, sn = itemgetter(
            'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(np.load(
                'results/CNMF-HRshapes' + batchname + '.npz'))
        A2 = A2.item()
    except:
        # ## run pipeline to determine shapes on high-res data
        t1 = time()
        Yr, sn, g, _ = cse.pre_processing.preprocess_data(
            Yr, **options['preprocess_params'])
        try:
            f_in, Ain, b_in, Cin = itemgetter(
                'f_in', 'Ain', 'b_in', 'Cin')(np.load('results/init.npz'))
        except:
            Atmp, Ctmp, b_in, f_in, center = cse.initialization.initialize_components(
                Y, **options['init_params'])
            print time() - t1
            print('DONE!')
            # refined by adding 'neurons'
            refine_components = True
            if refine_components:
                Cn = cse.utilities.local_correlations(Y)
                Ain, Cin = cse.utilities.manually_refine_components(
                    Y, options['init_params']['gSig'], coo_matrix(Atmp), Ctmp, Cn, thr=0.9)
            else:
                Ain, Cin = Atmp, Ctmp
            np.savez_compressed('results/init', Ain=Ain, Cin=Cin, b_in=b_in, f_in=f_in)

        # restict to small batch of the data
        if batch:
            Yr = Yr[:, :T / 2**batch]
            Cin = Cin[:, :T / 2**batch]
            f_in = f_in[:, :T / 2**batch]

        # update_spatial_components
        t1 = time()
        A, b, Cin = cse.spatial.update_spatial_components(
            Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])
        t_elSPATIAL = time() - t1
        print t_elSPATIAL
        print('DONE!')

        # update_temporal_components
        t1 = time()
        C, f, S, bl, c1, neurons_sn, g, YrA = cse.temporal.update_temporal_components(
            Yr, A, b, Cin, f_in, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
        t_elTEMPORAL2 = time() - t1
        print t_elTEMPORAL2
        print('DONE!')

        # merge components corresponding to the same neuron
        t1 = time()
        A_m, C_m, nr_m, merged_ROIs, S_m, bl_m, c1_m, sn_m, g_m = \
            cse.merging.merge_components(Yr, A, b, C, f, S, sn, options[
                'temporal_params'], options['spatial_params'], bl=bl, c1=c1,
                sn=neurons_sn, g=g, thr=0.8, fast_merge=True)
        t_elMERGE = time() - t1
        print t_elMERGE
        print('DONE!')

        # refine spatial components
        t1 = time()
        A2, b2, C2 = cse.spatial.update_spatial_components(
            Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
        print time() - t1
        print('DONE!')

        # normalize
        z = np.linalg.norm(A2.toarray(), 2, 0)
        A2 = coo_matrix(A2.toarray() / z)
        C2 = C2 * z.reshape(-1, 1)

        # save intermediate results
        np.savez_compressed('results/CNMF-HRshapes' + batchname + '.npz',
                            **{'A2': A2, 'b2': b2, 'C2': C2, 'f': f,
                               'A_m': A_m, 'C_m': C_m, 'sn': sn})
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
                    activity[mcell] += (A[mcell] - np.dot(B[mcell].T, activity)) / B[mcell, mcell]
                    activity[mcell][activity[mcell] < 0] = 0
            return activity
        Ab = coo_matrix(np.c_[A, b].T)
        Cf = np.r_[C, f.reshape(1, -1)]
        Cf = HALS4activity(np.reshape(Y, (d1 * d2, T)), Ab, Cf)
        return Cf[:-1], Cf[-1].reshape(1, -1)

    for shuffled in [False, True]:

        if shuffled:  # data, from 'true' generative model by shuffling residuals in time
            # ## load or generate shuffled data
            try:
                Yr = np.load('YrStratShuffled.npy', mmap_mode='r')
            except:
                Yr = np.load('Yr.npy', mmap_mode='r')
                # load shapes inferred on original high-res data
                A2, b2, C2, f, A_m, C_m, sn = itemgetter(
                    'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(
                    np.load('results/CNMF-HRshapes.npz'))
                A2 = A2.item()
                ssub1 = np.load('results/decimate.npz')['ssub'].item()[1]
                # ssub1 is tuple (C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA)
                residual = Yr - A2.dot(ssub1[0]).astype('float32') - \
                    b2.dot(ssub1[1]).astype('float32')
                np.random.seed(0)

                def stratified_reshuffle(signal, res):
                    return res[np.apply_along_axis(np.random.permutation, 1,
                                                   np.argsort(signal).reshape(200, 10))
                               .ravel()[np.argsort(np.argsort(signal))]]

                YrShuffled = Yr - residual
                # + np.apply_along_axis(np.random.permutation, 1, residual)
                YrShuffled += np.asarray([stratified_reshuffle(y, residual[i])
                                          for i, y in enumerate(YrShuffled)])
                np.save('YrStratShuffled', YrShuffled)
                del YrShuffled
                del residual
                Yr = np.load('YrStratShuffled.npy', mmap_mode='r')
            # ## get spatial components on high-resolution data
            try:  # load shapes inferred on shuffled high-res data
                A2, b2, C2, f, A_m, C_m, sn = itemgetter(
                    'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(
                    np.load('results/CNMF-HRshapes-stratshuffled' + batchname + '.npz'))
                A2 = A2.item()
            except:
                A2, b2, C2, f, A_m, C_m, sn = itemgetter(
                    'A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(
                    np.load('results/CNMF-HRshapes' + batchname + '.npz'))

                # restict to batch of the data
                if batch:
                    Yr = Yr[:, :T / 2**batch]
                    C_m = C_m[:, :T / 2**batch]
                    f = f[:, :T / 2**batch]

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
                np.savez_compressed('results/CNMF-HRshapes-stratshuffled' + batchname + '.npz',
                                    **{'A2': A2, 'b2': b2, 'C2': C2, 'f': f,
                                       'A_m': A_m, 'C_m': C_m, 'sn': sn})
                if batch:
                    Yr = np.load('YrStratShuffled.npy', mmap_mode='r')
            N = A2.shape[1]
        else:
            Yr = np.load('Yr.npy', mmap_mode='r')

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
        np.savez_compressed('results/decimate-stratshuffled' + batchname + '.npz' if shuffled
                            else 'results/decimate' + batchname + '.npz',
                            **{'ssub': ssub, 'ssubX': ssubX})

        #

        if not batch:

            # ## infer shapes on downsampled data

            # decimate along x and y
            t1 = time()
            A2 = {}
            b2 = {}
            C2 = {}
            for ds in dsls:
                tmp = downscale_local_mean(Yr.reshape(d1, d2, T), (ds, ds, 1))
                options['spatial_params']['dims'] = tmp.shape[:2]
                options['spatial_params']['n_pixels_per_process'] = \
                    options['preprocess_params']['n_pixels_per_process'] / ds**2
                A2[ds], b2[ds], C2[ds] = cse.spatial.update_spatial_components(
                    tmp.reshape(-1, T), C_m, f,
                    downscale_local_mean(np.ravel(A_m)[0].toarray().
                                         reshape(d1, d2, N), (ds, ds, 1)).reshape(-1, N),
                    sn=downscale_local_mean(sn.reshape(d1, d2), (ds, ds)).reshape(-1),
                    **options['spatial_params'])
                print time() - t1
            print('DONE!')
            options['spatial_params']['dims'] = d1, d2
            options['spatial_params']['n_pixels_per_process'] = \
                options['preprocess_params']['n_pixels_per_process']

            # decimate along x only
            t1 = time()
            A2X = {}
            b2X = {}
            C2X = {}
            for ds in dsls:
                tmp = downscale_local_mean(Yr.reshape(d1, d2, T), (ds, 1, 1))
                options['spatial_params']['dims'] = tmp.shape[:2]
                options['spatial_params']['n_pixels_per_process'] = \
                    options['preprocess_params']['n_pixels_per_process'] / ds
                A2X[ds], b2X[ds], C2X[ds] = cse.spatial.update_spatial_components(
                    tmp.reshape(-1, T), C_m, f,
                    downscale_local_mean(np.ravel(A_m)[0].toarray().
                                         reshape(d1, d2, N), (ds, 1, 1)).reshape(-1, N),
                    sn=downscale_local_mean(sn.reshape(d1, d2), (ds, 1)).ravel(),
                    **options['spatial_params'])
                print time() - t1
            print('DONE!')
            options['spatial_params']['dims'] = d1, d2
            options['spatial_params']['n_pixels_per_process'] = \
                options['preprocess_params']['n_pixels_per_process']

            # infer temporal components for various downsampling scenarios
            ssub = {}
            ssubX = {}
            t1 = time()
            for ds in dsls:
                C0, f0 = hals(downscale_local_mean(Yr.reshape(d1, d2, T),
                                                   (ds, ds, 1)), A2[ds].toarray(), b2[ds])
                ssub[ds] = cse.temporal.update_temporal_components(
                    downscale_local_mean(Yr.reshape(d1, d2, T), (ds, ds, 1)).reshape(-1, T),
                    A2[ds], b2[ds], C0, f0, bl=None, c1=None, sn=None, g=None,
                    **options['temporal_params'])
                C0, f0 = hals(downscale_local_mean(Yr.reshape(d1, d2, T),
                                                   (ds, 1, 1)), A2X[ds].toarray(), b2X[ds])
                ssubX[ds] = ssub[ds] if ds == 1 else cse.temporal.update_temporal_components(
                    downscale_local_mean(Yr.reshape(d1, d2, T), (ds, 1, 1)).reshape(-1, T),
                    A2X[ds], b2X[ds], C0, f0, bl=None, c1=None, sn=None, g=None,
                    **options['temporal_params'])
                print time() - t1
            print('DONE!')

            # save results
            np.savez_compressed('results/decimate-stratshuffled-LR.npz' if shuffled
                                else 'results/decimate-LR.npz',
                                **{'ssub': ssub, 'ssubX': ssubX})

            #

            # ## interleave

            # load shapes inferred on high-res data
            A2, b2, C2, f, A_m, C_m, sn = itemgetter('A2', 'b2', 'C2', 'f', 'A_m', 'C_m', 'sn')(
                np.load('results/CNMF-HRshapes-stratshuffled.npz' if shuffled
                        else 'results/CNMF-HRshapes.npz'))
            A2 = A2.item()
            N = A2.shape[1]

            # infer temporal components for various downsampling scenarios
            il = {}
            il2 = {}
            t1 = time()
            for ds in dsls:
                Y = np.zeros_like(Yr)
                Y[:, ::2] = cse.temporal.standardDownscale(
                    Yr[:, ::2].reshape(d1, d2, -1), ds).reshape(d1 * d2, -1)
                Y[:, 1::2] = cse.temporal.shiftedDownscale(
                    Yr[:, 1::2].reshape(d1, d2, -1), ds).reshape(d1 * d2, -1)
                C0, f0 = hals(Y.reshape(d1, d2, T), A2.toarray(), b2)
                il[ds] = cse.temporal.update_temporal_components_interleaved(
                    Y, A2, b2, C0, f0, dims=(d1, d2), ds=ds,
                    bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
                il2[ds] = cse.temporal.update_temporal_components_interleaved(
                    Y, A2, b2, C0, f0, dims=(d1, d2), ds=ds, bl=None, c1=None,
                    sn=None, g=None, interleave=True, **options['temporal_params'])
                print time() - t1
            print('DONE!')

            # save results
            np.savez_compressed('results/decimate-interleave-stratshuffled.npz' if shuffled
                                else 'results/decimate-interleave.npz', **{'il': il, 'il2': il2})


# STOP CLUSTER
cse.utilities.stop_server()
