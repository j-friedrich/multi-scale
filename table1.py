import numpy as np
from scipy.ndimage.filters import percentile_filter
from timeit import Timer
from CNMF import LocalNMF, OldLocalNMF
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper
from oasis import constrained_oasisAR1  # github.com/j-friedrich/OASIS


# Fetch Data
data = np.load('data_zebrafish.npy')
sig = (3, 3)
N = 46

runs = 10


# load data

t_data = [Timer(lambda: np.load('data_zebrafish.npy'))
          .timeit(1) for _ in range(runs)]


# decimate

data_dec = data.reshape((-1, 30) + data.shape[1:]).mean(1)
t_dec = [Timer(lambda: data.reshape((-1, 30) + data.shape[1:]).mean(1))
         .timeit(1) for _ in range(runs)]


# find neurons

centers = cse.greedyROI(data_dec.transpose(1, 2, 0), nr=N, gSig=[4, 4])[2].astype(int)
t_Greedy = [Timer(lambda: cse.greedyROI(data_dec.transpose(1, 2, 0), nr=N, gSig=[4, 4])[2])
            .timeit(1) for _ in range(runs)]
t_Greedy_old = [Timer(lambda: cse.greedyROI(data.transpose(1, 2, 0), nr=N, gSig=[4, 4])[2])
                .timeit(1) for _ in range(runs)]


# NMF

MSE = [LocalNMF(data, centers, sig, iters=5, mb=30, iters0=30,
                ds=[3, 3])[0] for _ in range(runs)]
# -2 cause last iteration updated shapes, leaving traces invariant
t_NMF = np.asarray(MSE)[:, -2, 0]
t_NMF_old = np.asarray([OldLocalNMF(data, centers, sig, iters=25)[0]
                        for _ in range(runs)])[:, -2, 0]


# dF/F

activity = LocalNMF(data, centers, sig, iters=5, mb=30, iters0=30, ds=[3, 3])[2]
b = np.apply_along_axis(lambda x: percentile_filter(
    x, 20, 300, mode='nearest'), 1, activity[:-1])
series = (activity[:-1] - b) / (b + 10)
series -= series.min(1).reshape(-1, 1)


def foo(activityQ):
    b = np.apply_along_axis(lambda x: percentile_filter(
        x, 20, 300, mode='nearest'), 1, activity[:-1])
    series = (activity[:-1] - b) / (b + 10)
    series -= series.min(1).reshape(-1, 1)
    return series

t_dF = [Timer(lambda: foo(activity)).timeit(1) for _ in range(runs)]


# denoise & deconvolve

def bar(s):
    tmp = cse.deconvolution.estimate_parameters(s, 1)
    constrained_oasisAR1(s, .97 * tmp[0][0], tmp[1], True)

t_OASIS = [Timer(lambda: map(bar, series.astype(float)))
           .timeit(1) for _ in range(runs)]

t_CVXPY = [Timer(lambda: map(lambda x: cse.deconvolution
                             .constrained_foopsi(x, p=1),
                             series.astype(float))).timeit(1)
           for _ in range(runs)]


# print table

print " load data   %6.3f+-%.3f   %.3f+-%.3f" % \
    tuple([np.mean(t_data), np.std(t_data) / np.sqrt(runs)] * 2)
print " decimate         N/A        %.3f+-%.3f" % \
    (np.mean(t_dec), np.std(t_dec) / np.sqrt(runs))
print "detect ROIs  %6.3f+-%.3f   %.3f+-%.3f" % \
    (np.mean(t_Greedy_old), np.std(t_Greedy_old) / np.sqrt(runs),
     np.mean(t_Greedy), np.std(t_Greedy) / np.sqrt(runs))
print "    NMF      %6.3f+-%.3f   %.3f+-%.3f" % \
    (np.mean(t_NMF_old), np.std(t_NMF_old) / np.sqrt(runs),
     np.mean(t_NMF), np.std(t_NMF) / np.sqrt(runs))
print "   dF/F      %6.3f+-%.3f   %.3f+-%.3f" % \
    tuple([np.mean(t_dF), np.std(t_dF) / np.sqrt(runs)] * 2)
print "  denoise    %6.3f+-%.3f   %.3f+-%.3f" % \
    (np.mean(t_CVXPY), np.std(t_CVXPY) / np.sqrt(runs),
     np.mean(t_OASIS), np.std(t_OASIS) / np.sqrt(runs))
