import numpy as np
from numpy import min, max, asarray, percentile, zeros, exp, ones, dot, where,\
    r_, ix_, arange, nan_to_num, prod, repeat, argsort, outer, clip
from time import time
from scipy.linalg import eigh


def HALS4activity(data, S, activity, iters=1, nonneg=True):
    A = S.dot(data.T)
    B = S.dot(S.T)
    for _ in range(iters):
        for ll in range(len(S)):
            activity[ll] += nan_to_num((A[ll] - dot(B[ll].T, activity)) / B[ll, ll])
            if nonneg:
                activity[ll][activity[ll] < 0] = 0
    return activity


def LocalNMF(data, centers, sig, iters=10, verbose=False, adaptBias=True,
             iters0=30, mb=30, ds=None, method=None, M=100, nonneg=True):
    """
    Parameters
    ----------
    data : array, shape (T, X, Y[, Z])
        block of the data
    centers : array, shape (L, D) or int
        if array : L centers of suspected neurons where D is spatial dimension (2 or 3)
        if int : initial number of randomly placed tiles, ~3-10 times number of neurons
    sig : array, shape (D,)
        size of the gaussian kernel in different spatial directions
    iters : int
        number of final iterations on whole data
    verbose : boolean
        print progress and record MSE if true (about 2x slower)
    adaptBias : boolean
        subtract rank 1 estimate of bias
    iters0 : int
        numbers of initial iterations on subset
    mb : int
        minibatchsize for temporal decimation
    ds : array, shape (D,)
        factor for spatial decimation in different spatial directions
    method : 'random', 'svd', 'subsample' or None
        compression method
    M : int
        compressed size
    nonneg: boolean
        if True, enforce nonnegative also on compressed data



    Returns
    -------
    MSE_array : list of pairs [t,mse]
        Time and Mean square error during algorithm operation
    shapes : array, shape (L+adaptBias, X, Y (,Z))
        the neuronal shape vectors
    activity : array, shape (L+adaptBias, T)
        the neuronal activity for each shape
    boxes : array, shape (L, D, 2)
        edges of the boxes in which each neuronal shapes lie
    """
    t = time()
    tsub = 0

    # Initialize Parameters
    dims = data.shape
    D = len(dims)
    R = (3 * asarray(sig)).astype('uint8')  # size of bounding box is 3 times size of neuron
    L = len(centers)
    mask = []  # binary matrix, indicates where shapes has non-zero entries
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    if iters0 == 0 or ds is None:
        ds = ones(D - 1, dtype='uint8')
    else:
        ds = asarray(ds, dtype='uint8')

### Function definitions ###
    def GetBox(centers, R, dims):
        D = len(R)
        box = zeros((D, 2), dtype=int)
        for dd in range(D):
            box[dd, 0] = max((centers[dd] - R[dd], 0))
            box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
        return box

    def HALS4shape(data, S, activity, iters=1):
        C = activity.dot(data)
        D = activity.dot(activity.T)
        for _ in range(iters):
            for ll in range(L + adaptBias):
                if ll == L:
                    S[ll] = clip(S[ll] +
                                 nan_to_num((C[ll] - dot(D[ll], S)) / D[ll, ll]), 0, np.inf)
                else:
                    S[ll, mask[ll]] = clip(S[ll, mask[ll]] +
                                           nan_to_num((C[ll, mask[ll]] -
                                                       dot(D[ll], S[:, mask[ll]])) / D[ll, ll]),
                                           0, np.inf)
        return S

    mse = lambda res: res.ravel().dot(res.ravel()) / res.size

### Initialize shapes (with boxes and mask), activity, and background ###
    if mb > 1:  # decimation
        # temporal downsampling
        data0 = data[:len(data) / mb * mb].reshape((-1, mb) + data.shape[1:])\
            .mean(1).astype('float32')
        # spatial downsampling
        if D == 4:
            data0 = data0.reshape(
                len(data0), dims[1] / ds[0], ds[0], dims[2] / ds[1], ds[1],
                dims[3] / ds[2], ds[2]).mean(2).mean(3).mean(4)
            activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1]),
                             map(int, centers[:, 2] / ds[2])].T
        else:
            data0 = data0.reshape(len(data0), dims[1] / ds[0],
                                  ds[0], dims[2] / ds[1], ds[1]).mean(2).mean(3)
            activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1])].T
        dims0 = data0.shape
        # reshape tensor to matrix
        data0 = data0.reshape(dims0[0], -1)
    elif iters0:  # some other compression
        data0 = data.reshape(dims[0], -1).astype('float32')
        dims0 = dims
    else:  # no compression
        if D == 4:
            activity = data[:, centers[:, 0], centers[:, 1], centers[:, 2]].T.astype('float32')
        else:
            activity = data[:, centers[:, 0], centers[:, 1]].T.astype('float32')
        dims0 = dims

    if method == 'subsample':
        data0 = data0[np.linspace(0, dims0[0] - 1, M).astype(int)]
        dims0 = (M,) + dims0[1:]

    elif method == 'random':
        np.random.seed(1)
        # Mariano Tepper, Guillermo Sapiro: COMPRESSED NONNEGATIVE MATRIX
        # FACTORIZATION IS FAST AND ACCURATE
        Om = np.random.randn(np.prod(dims0[1:]), M).astype('float32')
        # B = data0.dot(data0.T.dot(data0.dot(Om)))
        B = data0.dot(Om)
        Lmatrix = np.linalg.qr(B)[0]
        Om = np.random.randn(dims0[0], M).astype('float32')
        # B = data0.T.dot(data0.dot(data0.T.dot(Om)))
        B = data0.T.dot(Om)
        Rmatrix = np.linalg.qr(B)[0].T
        dataL = Lmatrix.T.dot(data0)
        dataR = data0.dot(Rmatrix.T)

    elif method == 'svd':
        if mb > 1:
            data_dec = data0.copy()
        COV = data0.dot(data0.T)
        _, V = eigh(COV, eigvals=(len(COV) - M, len(COV) - 1))
        data0 = V.T.dot(data0)
        dims0 = (M,) + dims0[1:]

    if method is not None:
        activity = data0.reshape(dims0)[:, centers[:, 0], centers[:, 1]].T

    data = data.astype('float32').reshape(dims[0], -1)
    # initialize shapes as Gaussians
    S = zeros((L + adaptBias, prod(dims0[1:])), dtype='float32')
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll] / ds, R / ds, dims0[1:])
        temp = zeros(dims0[1:])
        temp[map(lambda a: slice(*a), boxes[ll])]=1
        mask += where(temp.ravel())
        temp = [(arange(dims[i + 1] / ds[i]) - centers[ll][i] / float(ds[i])) ** 2 /
                (2 * (sig[i] / float(ds[i])) ** 2)
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        S[ll, mask[ll]] = temp.ravel()[mask[ll]]
    if adaptBias:
        # Initialize background as 20% percentile
        if method == 'svd':
            activity = r_[activity, V.sum(0).reshape(1, -1)]
            S[-1] = percentile(data_dec, 20, 0) if mb > 1 else percentile(data, 20, 0)
        else:
            activity = r_[activity, ones((1, dims0[0]), dtype='float32')]
            S[-1] = percentile(data0, 20, 0) if mb > 0 else percentile(data, 20, 0)

### Get shape estimates on subset of data ###
    if iters0:
        for kk in range(iters0):
            if method == 'random':
                S = HALS4shape(dataL, S, activity.dot(Lmatrix))
                activity = HALS4activity(dataR, S.dot(Rmatrix.T), activity, nonneg=nonneg)
            else:
                S = HALS4shape(data0, S, activity)
                activity = HALS4activity(data0, S, activity, nonneg=nonneg)

    ### Back to full data ##
        activity = ones((L + adaptBias, dims[0]),
                        dtype='float32') * activity.mean(1).reshape(-1, 1)
        if D == 4:
            S = repeat(repeat(repeat(S.reshape((-1,) + dims0[1:]),
                                     ds[0], 1), ds[1], 2), ds[2], 3).reshape(L + adaptBias, -1)
        else:
            S = repeat(repeat(S.reshape((-1,) + dims0[1:]),
                              ds[0], 1), ds[1], 2).reshape(L + adaptBias, -1)
        for ll in range(L):
            boxes[ll] = GetBox(centers[ll], R, dims[1:])
            # boxes[ll] *= ds.reshape(-1, 1) # this potentially yields a bigger box
            temp = zeros(dims[1:])
            temp[map(lambda a: slice(*a), boxes[ll])] = 1
            mask[ll] = asarray(where(temp.ravel())[0])

        # from now on more iterations cause initial dot product in HALS is expensive for full data
        activity = HALS4activity(data, S, activity, 7)
        tsub += time()
        residual = data - activity.T.dot(S)
        MSE = mse(residual)
        tsub -= time()
        MSE_array += [[time() - t + tsub, MSE]]

        S = HALS4shape(data, S, activity, 7)
        tsub += time()
        residual = data - activity.T.dot(S)
        MSE = mse(residual)
        tsub -= time()
        MSE_array += [[time() - t + tsub, MSE]]

#### Main Loop ####
    for kk in range(iters):
        activity = HALS4activity(data, S, activity, 10)
        tsub += time()
        residual = data - activity.T.dot(S)
        MSE = mse(residual)
        tsub -= time()
        MSE_array += [[time() - t + tsub, MSE]]

        S = HALS4shape(data, S, activity, 10)
        tsub += time()
        residual = data - activity.T.dot(S)
        MSE = mse(residual)
        tsub -= time()
        MSE_array += [[time() - t + tsub, MSE]]

        if verbose:
            print('{0:1d}: MSE = {1:.5f}'.format(kk, MSE))
            if kk == (iters - 1):
                print('Maximum iteration limit reached')

    return asarray(MSE_array), S.reshape((-1,) + dims[1:]), activity, boxes


def OldLocalNMF(data, centers, sig, NonNegative=True, iters=10, verbose=False, adaptBias=True):
    """
    Parameters
    ----------
    data : array, shape (T, X, Y[, Z])
        block of the data
    centers : array, shape (L, D)
        L centers of suspected neurons where D is spatial dimension (2 or 3)
    activity : array, shape (L, T)
        traces of temporal activity
    sig : array, shape (D,)
        size of the gaussian kernel in different spatial directions
    NonNegative : boolean
        if True, neurons should be considered as non-negative
    iters : int
        maximum number of iterations
    verbose : boolean
        print progress if true
    adaptBias : boolean
        subtract rank 1 estimate of bias

    Returns
    -------
    MSE_array : list
        Mean square error during algorithm operation
    shapes : list (length L) of lists (var length)
        the neuronal shape vectors
    activity : array, shape (L, T)
        the neuronal activity for each shape
    boxes : array, shape (L, D, 2)
        edges of the boxes in which each neuronal shapes lie
    """
    t = time()

    # Initialize Parameters
    dims = data.shape
    D = len(dims)
    R = 3 * asarray(sig)  # size of bounding box is 3 times size of neuron
    L = len(centers)
    shapes = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    activity = zeros((L, dims[0]))

### Function definitions ###
    def GetBox(centers, R, dims):
        D = len(R)
        box = zeros((D, 2), dtype=int)
        for dd in range(D):
            box[dd, 0] = max((centers[dd] - R[dd], 0))
            box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
        return box

    def RegionAdd(Z, X, box):
        # Parameters
        #  Z : array, shape (T, X, Y[, Z]), dataset
        #  box : array, shape (D, 2), array defining spatial box to put X in
        #  X : array, shape (T, prod(diff(box,1))), Input
        # Returns
        #  Z : array, shape (T, X, Y[, Z]), Z+X on box region
        Z[[slice(len(Z))] + list(map(lambda a: slice(*a), box))
          ] += X.reshape((r_[-1, box[:, 1] - box[:, 0]]))
        return Z

    def RegionCut(X, box):
        # Parameters
        #  X : array, shape (T, X, Y[, Z])
        #  box : array, shape (D, 2), region to cut
        # Returns
        #  res : array, shape (T, prod(diff(box,1))),
        dims = X.shape
        return X[[slice(dims[0])] + list(map(lambda a: slice(*a), box))].reshape((dims[0], -1))

    mse = lambda res: res.ravel().dot(res.ravel()) / res.size

# Initialize shapes, activity, and residual
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll], R, dims[1:])
        temp = [(arange(dims[i + 1]) - centers[ll][i]) ** 2 / (2 * sig[i] ** 2)
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        temp.shape = (1,) + dims[1:]
        temp = RegionCut(temp, boxes[ll])
        shapes.append(temp[0])
    residual = data.astype('float')
    if adaptBias:
        # Initialize background as 30% percentile
        b_s = percentile(residual, 30, 0)
        residual -= b_s
    # Initialize activity from strongest to weakest
    # based on data-background-stronger neurons and Gaussian shapes
    for ll in argsort([residual[:, c[0], c[1]].max() for c in centers])[::-1]:
        X = RegionCut(residual, boxes[ll])
        activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
        if NonNegative:
            activity[ll][activity[ll] < 0] = 0
        residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

    # (Re)calculate background based on data-neurons using nonnegative greedy PCA
    if adaptBias:
        residual += b_s
        residual.shape = (dims[0], -1)
        b_s = b_s.ravel()
        b_t = dot(residual, b_s) / dot(b_s, b_s)
        b_t[b_t < 0] = 0
        b_s = dot(residual.T, b_t) / dot(b_t, b_t)
        b_s[b_s < 0] = 0
        residual -= outer(b_t, b_s)
        residual.shape = dims
        zz = b_t.mean()
        b_s *= zz
        b_t /= zz

    tsub = time()
    MSE = mse(residual)
    tsub -= time()
    MSE_array = [[time() + tsub - t, MSE]]

#### Main Loop ####
    for kk in range(iters):
        for ll in range(L):
            # cut region and add neuron
            as0 = outer(activity[ll], shapes[ll])
            X = RegionCut(residual, boxes[ll]) + as0
            # NonNegative greedy PCA
            for ii in range(3):
                activity[ll] = nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
                shapes[ll] = nan_to_num(dot(X.T, activity[ll]) / dot(activity[ll], activity[ll]))
                if NonNegative:
                    shapes[ll][shapes[ll] < 0] = 0
            as0 -= outer(activity[ll], shapes[ll])
            # Update region
            residual = RegionAdd(residual, as0, boxes[ll])

        # Recalculate background
        if adaptBias:
            residual.shape = (dims[0], -1)
            residual += outer(b_t, b_s)
            for _ in range(1):
                b_s = dot(residual.T, b_t) / dot(b_t, b_t)
                b_s[b_s < 0] = 0
                b_t = dot(residual, b_s) / dot(b_s, b_s)
                b_t[b_t < 0] = 0
            residual -= outer(b_t, b_s)
            residual.shape = dims

        # Measure MSE
        tsub = time()
        MSE = mse(residual)
        tsub -= time()
        MSE_array += [[time() + tsub - t, MSE]]
        if verbose:
            print('{0:1d}: MSE = {1:.3f}'.format(kk, MSE))
        if kk == (iters - 1):
            print('Maximum iteration limit reached')
    return MSE_array, shapes, activity, boxes


#####################################################################################
#
# example
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from numpy.random import randn, randint
    from scipy.ndimage.filters import gaussian_filter

    plt.rc('patch', lw=3)

    T = 300  # duration of the simulation
    sz = (150, 100)  # size of image
    sig = (5, 5)  # neurons size
    foo = 0.1 * randn(*((T,) + sz))  # noise
    bar = np.zeros((T,) + sz)
    N = 15  # number of neurons
    centers = []
    for i in range(N):
        centers += [[randint(x) for x in sz]]
        for j in range(T):
            bar[(j,) + tuple(centers[-1])] = abs(randn())
    data = 1 + foo + 10 * gaussian_filter(bar, (0,) + sig)
    centers = np.asarray(centers)
    MSE_array, shapes, activity, boxes = LocalNMF(data, centers, sig, verbose=True)

    denoised_data = activity[:N].T.dot(shapes[:N].reshape(N, -1)).reshape(data.shape)
    residual = data - activity.T.dot(shapes.reshape(len(shapes), -1)).reshape(data.shape)

    # Plot Results
    fig = plt.figure()
    plt.plot(*MSE_array.T)
    plt.xlabel('Time [s]')
    plt.ylabel('MSE')
    plt.show()

    plt.figure(figsize=(10, 5. * data.shape[1] / data.shape[2]))
    ax = plt.subplot(121)
    ax.scatter(*centers.T[::-1], s=40, marker='x', c='g')
    ax.set_title('Data + centers')
    ax.imshow(np.percentile(data, 98, axis=0), cmap='hot')
    ax2 = plt.subplot(122)
    ax2.scatter(*centers.T[::-1], s=40, marker='x', c='g')
    ax2.imshow(np.percentile(denoised_data, 98, axis=0), cmap='hot')
    ax2.set_title('Denoised data')
    plt.show()

    plt.figure(figsize=(15, 10))
    for i in range(N):
        plt.subplot(3, 5, i + 1)
        plt.imshow(shapes[i][map(lambda a: slice(*a), boxes[i])])
    plt.suptitle('Inferred shapes')
    plt.show()

    # Video Results
    fig = plt.figure(figsize=(12, 4. * data.shape[1] / data.shape[2]))
    ii = 0
    ax = plt.subplot(131)
    ax.scatter(*centers.T[::-1], s=40, marker='x', c='g')
    im = ax.imshow(data[ii], vmin=data.min(), vmax=data.max(), cmap='hot')
    ax.set_title('Data + centers')
    ax3 = plt.subplot(132)
    ax3.scatter(*centers.T[::-1], s=40, marker='x', c='g')
    im3 = ax3.imshow(denoised_data[ii], vmin=denoised_data.min(),
                     vmax=denoised_data.max(), cmap='hot')
    ax3.set_title('Denoised')
    ax2 = plt.subplot(133)
    ax2.scatter(*centers.T[::-1], s=40, marker='x', c='g')
    im2 = ax2.imshow(residual[ii], vmin=-residual.max(), vmax=residual.max(), cmap='seismic')
    ax2.set_title('Residual')

    def update(ii):
        im.set_data(data[ii])
        im2.set_data(residual[ii])
        im3.set_data(denoised_data[ii])
    ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, interval=30,
                                  repeat=False)
    plt.show()
