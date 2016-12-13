import numpy as np
import matplotlib.pyplot as plt
from CNMF import LocalNMF, OldLocalNMF
from functions import init_fig, simpleaxis
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper

init_fig()
plt.rc('lines', lw=3)
save_figs = False  # subfolder fig must exist if set to True

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
MSE_order = np.asarray(
    [OldLocalNMF(data, centers, sig, verbose=True, iters=iters)[0] for _ in range(runs)])

# no decimation
MSE_nodec = np.asarray([LocalNMF(data, centers, sig, verbose=True, iters=iters,
                                 iters0=0, mb=1)[0] for _ in range(runs)])

# decimate only in time
mbls = [2, 5, 10, 20, 30, 40]
MSE_decT = np.asarray(
    [[LocalNMF(data, centers, sig, verbose=True, iters=iters,
               iters0=30, mb=i, ds=[1, 1])[0]
      for i in mbls] for _ in range(runs)])

# decimate in time and space
MSE_decTS = np.asarray(
    [[LocalNMF(data, centers, sig, verbose=True, iters=iters,
               iters0=30, mb=30, ds=[i, i])[0]
      for i in [2, 3, 4, 6, 8]] for _ in range(runs)])


# ### plots ###
z = MSE_nodec[-1, -1, -1]


def adjust():
    plt.xticks(*[[0, 5, 10]] * 2)
    plt.yticks(*[[1.00, 1.01]] * 2)
    plt.xlim(0, 9)
    plt.ylim(75 / z, 76.05 / z)
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
plt.yticks(*[[1.00, 1.05]] * 2)
plt.xlim(0, 30)
plt.ylim(75 / z, 1.052)
plt.plot((0, 9, 9), (76.05 / z, 76.05 / z, 75 / z), 'k--')
if save_figs:
    plt.savefig('fig/MSE_order.pdf')
plt.show()

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
if save_figs:
    plt.savefig('fig/MSE_decT.pdf')
plt.show()

# spatial decimation
plt.figure()
plt.plot(*np.min([m[3] for m in MSE_decT], 0).T / np.array([[1], [z]]), label='1x1', c=col[3])
for k, i in enumerate(['2x2', '3x3', '4x4', '6x6']):
    line, = plt.plot(*np.min([m[k] for m in MSE_decTS], 0).T / np.array([[1], [z]]),
                     ls=['--', ':', '-.', ':'][k], label=i, c=col[(4 + 2 * k) % 8])
    if k == 3:
        line.set_dashes([8, 4, 2, 4, 2, 4])
lg = plt.legend(title='decimation factor', ncol=2, columnspacing=1, bbox_to_anchor=(1, 1))
adjust()
if save_figs:
    plt.savefig('fig/MSE_decT+S.pdf')
plt.show()
