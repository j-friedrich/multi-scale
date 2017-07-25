import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile
from skimage.filters import gaussian
from scipy.ndimage.measurements import center_of_mass
from functions import init_fig, simpleaxis, noclip, gfp, showpause
import ca_source_extraction as cse  # https://github.com/j-friedrich/CaImAn/tree/multi-scale_paper

if matplotlib.__version__[0] == '2':
    matplotlib.style.use('classic')
    
try:
    from sys import argv
    from os.path import isdir
    figpath = argv[1] if isdir(argv[1]) else False
except:
    figpath = False

init_fig()

plt.rc('font', size=20, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})


# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
[vermillon, orange, yellow, green, cyan, blue, purple, grey] = col

#

# Load data

dsls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

d1, d2, T = (512, 512, 2000)
filename = '180um_20fps_350umX350um.tif'
Yr = tifffile.TiffFile(filename).asarray().astype(dtype='float32')
Yr = np.transpose(Yr, (1, 2, 0))
Ymax = Yr.max(-1)
Yr = np.reshape(Yr, (d1 * d2, T), order='F')
A2 = np.load('results/CNMF-HRshapes.npz')['A2'].item()
N = A2.shape[1]

ssub = np.load('results/decimate.npz')['ssub'].item()
trueC = ssub[1][0]
trueSpikes = ssub[1][2]

A_or, C_or, srt = cse.utilities.order_components(A2, ssub[1][0])
A_or = np.transpose([gaussian(a.reshape(d1, d2), 1).ravel() for a in A_or.T])

#

#

# Plot data

fig = plt.figure(figsize=(7.1, 6))
A2 = np.load('results/CNMF-HRshapes.npz')['A2'].item()
A2 = np.transpose([gaussian(a.reshape(d1, d2), 1).ravel() for a in A2.toarray().T])
tmp = plt.rcParams['lines.linewidth']
plt.rcParams['lines.linewidth'] = 4
cse.utilities.plot_contours(A2, Yr.reshape(-1, T / 20, 20).mean(-1).max(-1).reshape(d1, d2).T,
                            thr=0.9, display_numbers=False, colors=orange)
plt.rcParams['lines.linewidth'] = tmp
A2 = np.load('results/CNMF-HRshapes-Aon1stHalf.npz')['A2'].item()
A2 = np.transpose([gaussian(a.reshape(d1, d2), 1).ravel() for a in A2.toarray().T])
cse.utilities.plot_contours(A2, Yr.reshape(-1, T / 20, 20).mean(-1).max(-1).reshape(d1, d2).T,
                            thr=0.9, display_numbers=False, colors=blue)
ax = plt.gca()
im = ax.imshow(Yr[:, :T / 2].reshape(-1, T / 2 / 20, 20).mean(-1).max(-1).reshape(d1, d2).T,
               cmap=gfp, vmin=400, vmax=4000)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, label='Maximum Projection')
cb.set_ticks(range(1000, 4000, 1000))
plt.yticks(range(1000, 4000, 1000), range(1000, 4000, 1000), fontsize=16)
cb.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
plt.subplots_adjust(.005, 0, .89, 1)
plt.savefig(figpath + '/data-1stHalf.pdf', bbox_inches='tight',
            pad_inches=.01) if figpath else showpause()


#

# # correlations

init_fig()

# find for each neuron obtianed on the full data the idx of the corresponding neuron
# obtained on small batch of the data, if it exists otherwise set idx to None
A2 = np.load('results/CNMF-HRshapes.npz')['A2'].item()
centers = np.asarray([center_of_mass(a.reshape(d1, d2)) for a in A2.toarray().T])

A2h = np.load('results/CNMF-HRshapes-Aon1stHalf.npz')['A2'].item()
centersH = np.asarray([center_of_mass(a.reshape(d1, d2)) for a in A2h.toarray().T])
# distances between all neuron centers
dist = [[np.linalg.norm(cH - c) for cH in centersH] for c in centers]
# distribution of minimal distances, large distance indicates removed neuron
print np.sort(np.min(dist, 1))
# map to closest neuron
idx = np.argmin(dist, 1)
# number of neurons mapped to same target neurons cause one of them has
# been removed on small batch of the data and hence misses a partner
print len(np.argmin(dist, 1)) - len(set(np.argmin(dist, 1)))
# indices of those targets
idx2 = np.sort(idx)[np.where(np.sort(idx)[1:] - np.sort(idx)[:-1] == 0)]
# distances to target neurons
print[np.min(dist, 1)[np.argmin(dist, 1) == i] for i in idx2]
# the ones far away have been removed on decimated data, hence set idx to None
idxH = np.asarray([None if np.min(dist, 1)[k] > 7 else idx[k] for k in range(len(idx))])

A2q = np.load('results/CNMF-HRshapes-Aon1stQuarter.npz')['A2'].item()
centersQ = np.asarray([center_of_mass(a.reshape(d1, d2)) for a in A2q.toarray().T])
# distances between all neuron centers
dist = [[np.linalg.norm(cQ - c) for cQ in centersQ] for c in centers]
# distribution of minimal distances, large distance indicates removed neuron
print np.sort(np.min(dist, 1))
# map to closest neuron
idx = np.argmin(dist, 1)
# number of neurons mapped to same target neurons cause one of them has
# been removed on small batch of the data and hence misses a partner
print len(np.argmin(dist, 1)) - len(set(np.argmin(dist, 1)))
# indices of those targets
idx2 = np.sort(idx)[np.where(np.sort(idx)[1:] - np.sort(idx)[:-1] == 0)]
# distances to target neurons
print[np.min(dist, 1)[np.argmin(dist, 1) == i] for i in idx2]
# the ones far away have been removed on decimated data, hence set idx to None
idxQ = np.asarray([None if np.min(dist, 1)[k] > 7 else idx[k] for k in range(len(idx))])


def plotCorr(ssubs, compare, rng=slice(0, T), ca_or_spikes='ca',
             labels=None, title='Shapes obtained on', colors=[orange, blue, vermillon], ls='-'):
    """ map based on spatial distance, removed neurons are set to corr=0 """
    def foo(ssub, comp, idx=None):
        N, T = comp.shape
        cor = np.zeros((N, len(dsls)))
        for i, ds in enumerate(dsls):
            if len(ssub[ds][0]) == len(comp):
                cor[:, i] = np.nan_to_num(
                    [np.corrcoef(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n, rng],
                                 comp[n, rng])[0, 1] for n in range(N)])
            else:  # necessary if component has been removed on small batch
                for n, k in enumerate(idx):
                    if k is not None:  # component removed -> keep corr=0
                        cor[n, i] = np.nan_to_num(
                            np.corrcoef(ssub[ds][0 if ca_or_spikes == 'ca' else 2][k, rng],
                                        comp[n, rng])[0, 1])
        return cor

    for i, ssub in enumerate(ssubs):
        cor = foo(ssub, compare, [None, idxH, idxQ][i])
        plt.plot(dsls, np.mean(cor, 0), ls=ls, lw=4, c=colors[i],
                 label=None if labels is None else labels[i], clip_on=False)
        noclip(plt.errorbar(dsls, np.mean(cor, 0), yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                            lw=3, capthick=2, fmt='o', c=colors[i], clip_on=False))
    plt.xlabel('Spatial decimation')
    simpleaxis(plt.gca())
    plt.xticks(dsls, ['1', '', '', '', '', '8x8', '', '16x16', '24x24', '32x32'])
    plt.yticks(*[np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.2)] * 2)
    plt.xlim(dsls[0], dsls[-1])
    plt.ylim(.25, 1)
    plt.ylabel('Correlation w/ undecimated $C_1$/$S_1$', y=.42, labelpad=1)
    plt.subplots_adjust(.13, .15, .94, .96)
    if labels is not None:
        lg = plt.legend(loc=(-.01, -.01), title=title)
        return lg


ssub = np.load('results/decimate.npz')['ssub'].item()
trueC = ssub[1][0]
trueSpikes = ssub[1][2]
ssubH = np.load('results/decimate-Aon1stHalf.npz')['ssub'].item()
ssubQ = np.load('results/decimate-Aon1stQuarter.npz')['ssub'].item()

rng = slice(T / 2, T)
plt.figure()
first_legend = plotCorr([ssub, ssubH, ssubQ], trueC, rng, ca_or_spikes='ca',
                        labels=['full data', '1st half of data', '1st quarter of data'])
first_legend.get_title().set_position((-30, 0))
plotCorr([ssub, ssubH, ssubQ], trueSpikes, rng, ca_or_spikes='spikes', ls='--')
l1, = plt.plot([0, 1], [-1, -1], lw=4, c='k', label='denoised')
l2, = plt.plot([0, 1], [-1, -1], lw=4, c='k', ls='--', label='deconvolved')
ax = plt.gca().add_artist(first_legend)  # Add the legend manually to the current Axes.
plt.legend(handles=[l1, l2], loc=(.55, .78))  # Create another legend for the second line.
plt.savefig(figpath + '/Corr-AonSmallBatch.pdf') if figpath else showpause()


# ## shuffled data

ssub = np.load('results/decimate-stratshuffled.npz')['ssub'].item()
ssubH = np.load('results/decimate-stratshuffled-Aon1stHalf.npz')['ssub'].item()
ssubQ = np.load('results/decimate-stratshuffled-Aon1stQuarter.npz')['ssub'].item()

rng = slice(T / 2, T)
plt.figure()
first_legend = plotCorr([ssub, ssubH, ssubQ], trueC, rng, ca_or_spikes='ca')
plotCorr([ssub, ssubH, ssubQ], trueSpikes, rng, ca_or_spikes='spikes', ls='--')
plt.ylabel(r'Correlation w/ ground truth $C$\textsuperscript{s}/$S$\textsuperscript{s}',
           y=.42, labelpad=3)
plt.savefig(figpath + '/Corr-stratshuffled-AonSmallBatch.pdf') if figpath else plt.show(block=True)
