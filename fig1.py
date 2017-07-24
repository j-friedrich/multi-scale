import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CNMF import LocalNMF, OldLocalNMF
from functions import init_fig, simpleaxis, showpause
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper

if matplotlib.__version__[0] == '2':
    matplotlib.style.use('classic')

try:
    from sys import argv
    from os.path import isdir
    figpath = argv[1] if isdir(argv[1]) else False
except:
    figpath = False

init_fig()
plt.rc('lines', lw=3)

# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
# vermillon, orange, yellow, green, cyan, blue, purple, grey


# load data
data = np.load('data_zebrafish.npy')
sig = (3, 3)
# find neurons greedily
N = 46
centers = cse.greedyROI(data.reshape((-1, 30) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[4, 4])[2].astype(int)


# ### run NMF ###
runs = 5
iters = 100

# outer loop over neurons
try:
    MSE_order = np.load('results/MSE_order.npy')
except:
    MSE_order = np.asarray(
        [OldLocalNMF(data, centers, sig, verbose=True, iters=iters)[0] for _ in range(runs)])
    np.save('results/MSE_order.npy', MSE_order)

# no decimation
try:
    MSE_nodec = np.load('results/MSE_nodec.npy')
except:
    MSE_nodec = np.asarray([LocalNMF(data, centers, sig, verbose=True, iters=iters,
                                     iters0=0, mb=0)[0] for _ in range(runs)])
    np.save('results/MSE_nodec.npy', MSE_nodec)

# decimate only in time
mbls = [2, 5, 10, 20, 30, 40]
try:
    MSE_decT = np.load('results/MSE_decT.npy')
except:
    MSE_decT = np.asarray(
        [[LocalNMF(data, centers, sig, verbose=True, iters=iters,
                   iters0=30, mb=i, ds=[1, 1])[0]
          for i in mbls] for _ in range(runs)])
    np.save('results/MSE_decT.npy', MSE_decT)

# decimate in time and space
try:
    MSE_decTS = np.load('results/MSE_decT+S.npy')
except:
    MSE_decTS = np.asarray(
        [[LocalNMF(data, centers, sig, verbose=True, iters=iters,
                   iters0=30, mb=30, ds=[i, i])[0]
          for i in [2, 3, 4, 6, 8]] for _ in range(runs)])
    np.save('results/MSE_decT+S.npy', MSE_decTS)

# subsample
try:
    MSE_sub = np.load('results/MSE_sub.npy')
except:
    MSE_sub = np.asarray(
        [LocalNMF(data, centers, sig, verbose=True, iters=iters, mb=1, method='subsample')[0]
         for _ in range(runs)])
    np.save('results/MSE_sub.npy', MSE_sub)

# svd
try:
    MSE_svd = np.load('results/MSE_svd.npy')
except:
    MSE_svd = np.asarray(
        [LocalNMF(data, centers, sig, verbose=True, iters=iters, iters0=10, mb=1, method='svd')[0]
         for _ in range(runs)])
    np.save('results/MSE_svd.npy', MSE_svd)

# svd without enforcing nonneg activities
try:
    MSE_negsvd = np.load('results/MSE_negsvd.npy')
except:
    MSE_negsvd = np.asarray(
        [LocalNMF(data, centers, sig, verbose=True, iters=iters,
                  iters0=10, mb=1, method='svd', nonneg=False)[0] for _ in range(runs)])
    np.save('results/MSE_negsvd.npy', MSE_negsvd)

# random
try:
    MSE_random = np.load('results/MSE_random.npy')
except:
    MSE_random = np.asarray(
        [LocalNMF(data, centers, sig, verbose=True, iters=iters,
                  iters0=10, mb=1, method='random')[0] for _ in range(runs)])
    np.save('results/MSE_random.npy', MSE_random)

# decimate + svd
try:
    MSE_decsvd = np.load('results/MSE_dec+svd.npy')
except:
    MSE_decsvd = np.asarray(
        [LocalNMF(data, centers, sig, verbose=True, iters=iters,
                  iters0=10, mb=30, method='svd', M=60)[0] for _ in range(runs)])
    np.save('results/MSE_dec+svd.npy', MSE_decsvd)

# decimate + svd without enforcing nonneg activities
try:
    MSE_decnegsvd = np.load('results/MSE_dec+negsvd.npy')
except:
    MSE_decnegsvd = np.asarray(
        [LocalNMF(data, centers, sig, verbose=True, iters=iters,
                  iters0=10, mb=30, method='svd', nonneg=False, M=60)[0] for _ in range(runs)])
    np.save('results/MSE_dec+negsvd.npy', MSE_decnegsvd)

# decimate + random
try:
    MSE_decrandom = np.load('results/MSE_dec+random.npy')
except:
    MSE_decrandom = np.asarray(
        [LocalNMF(data, centers, sig, verbose=True, iters=iters,
                  iters0=10, mb=30, method='random', M=60)[0] for _ in range(runs)])
    np.save('results/MSE_dec+random.npy', MSE_decrandom)


# ### plots ###
z = MSE_nodec[-1, -1, -1]
l, u = .9965, 1.0105


def adjust():
    plt.xticks(*[[0, 5, 10]] * 2)
    plt.yticks(*[[1, 1.01], ['1.00', 1.01]])
    plt.xlim(0, 8.5)
    plt.ylim(l, u)
    plt.xlabel('Wall time [s]')
    simpleaxis(plt.gca())
    plt.subplots_adjust(.1, .155, .99, .99)


# update order
plt.figure()
plt.plot(*np.min(MSE_order, 0).T / np.array([[1], [z]]),
         label='one neuron at a time\n (HALS)', c='k')
plt.plot(*np.min(MSE_nodec, 0).T / np.array([[1], [z]]),
         label='one factor at a time\n (fast HALS)', c=col[0])
lg = plt.legend(bbox_to_anchor=(1, 1))
adjust()
plt.ylabel('normalized MSE', labelpad=-25)
plt.xticks(*[[0, 10, 20]] * 2)
plt.yticks(*[[1, 1.05], ['1.00', 1.05]])
plt.xlim(0, 30)
plt.ylim(l, 1.052)
plt.plot((0, 8.5, 8.5), (u, u, l), 'k--')
plt.savefig(figpath + '/MSE_order.pdf') if figpath else showpause()


# temporal decimation
plt.figure()
plt.plot(*np.min(MSE_nodec, 0).T / np.array([[1], [z]]), label='1', c=col[0])
for k, i in enumerate(mbls[1:]):
    plt.plot(*np.min([m[k + 1]
                      for m in MSE_decT], 0).T / np.array([[1], [z]]), label=i, c=col[k + 1])
lg = plt.legend(title='decimation factor', ncol=2, columnspacing=1, bbox_to_anchor=(1, 1))
for i, t in enumerate(lg.get_texts()):  # right align
    t.set_ha('right')
    t.set_position((25 + i / 3 * 10, 0))
adjust()
plt.ylabel('normalized MSE', labelpad=-25, y=.58)
plt.savefig(figpath + '/MSE_decT.pdf') if figpath else showpause()


# spatial decimation
plt.figure()
plt.plot(*np.min([m[3] for m in MSE_decT], 0).T / np.array([[1], [z]]), label='1x1', c=col[4])
for k, i in enumerate(['2x2', '3x3', '4x4', '6x6']):
    line, = plt.plot(*np.min([m[k] for m in MSE_decTS], 0).T / np.array([[1], [z]]),
                     ls=['--', ':', '-.', ':'][k], label=i, c=col[[3, 6, 0, 2][k]])
    if k == 3:
        line.set_dashes([8, 4, 2, 4, 2, 4])
lg = plt.legend(title='decimation factor', ncol=2, columnspacing=1, bbox_to_anchor=(1, 1))
adjust()
plt.savefig(figpath + '/MSE_decT+S.pdf') if figpath else showpause()


# other compression schemes
plt.figure()
for k, (d, label) in enumerate([(MSE_nodec, 'none'), (MSE_decT[:, 4], 'decimate'),
                                (MSE_random, 'random'), (MSE_negsvd, 'svd'),
                                (MSE_decrandom, 'dec.+random'), (MSE_decnegsvd, 'dec.+svd')]):
    plt.plot(*np.min(d, 0).T / np.array([[1], [z]]), label=label, c=col[[0, 4, 7, 6, 7, 6][k]],
             ls=['-', '--'][k // 4])
lg = plt.legend(bbox_to_anchor=(1.05, 1.05))
adjust()
plt.savefig(figpath + '/MSE_other.pdf') if figpath else plt.show(block=True)
