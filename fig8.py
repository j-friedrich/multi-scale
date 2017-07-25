import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile
from skimage.filters import gaussian
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

# Plot decimated data with interleaving grid

fig = plt.figure(figsize=(7.1, 6))
crd = cse.utilities.plot_contours(A_or, Ymax, thr=0.9, numbercolor='w', colors='w')
ax = plt.gca()
im = ax.imshow(Yr.reshape(d1, d2, T / 10, 10).mean(-1).max(-1).T, cmap=gfp, vmin=400, vmax=4000)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
plt.hlines(range(16, d2, 16), 0, d2, cyan)
plt.vlines(range(16, d1, 16), 0, d1, cyan)
plt.hlines(range(8, d2, 16), 0, d2, orange)
plt.vlines(range(8, d1, 16), 0, d1, orange)
plt.xlim(112, 288)
plt.ylim(128, 304)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, label='Maximum Projection')
cb.set_ticks(range(1000, 4000, 1000))
plt.yticks(range(1000, 4000, 1000), range(1000, 4000, 1000), fontsize=16)
cb.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
plt.subplots_adjust(.005, 0, .89, 1)
plt.savefig(figpath + '/data-interleave.pdf') if figpath else showpause()
#

#

# plot correlations

init_fig()


def plotCorr(ssubs, compare, dsls, labels=None,
             colors=[cyan, orange, green], ca_or_spikes='ca', ls='-'):
    def foo(ssub, comp):
        N, T = comp.shape
        cor = np.zeros((N, len(dsls)))
        for i, ds in enumerate(dsls):
            if len(ssub[ds][0]) == len(comp):
                cor[:, i] = np.nan_to_num(
                    [np.corrcoef(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                                 comp[n])[0, 1] for n in range(N)])
            else:  # necessary if update_spatial_components removed a component
                mapIdx = [np.argmax([np.corrcoef(s, tC)[0, 1] for tC in comp])
                          for s in ssub[ds][0 if ca_or_spikes == 'ca' else 2]]
                for n in range(len(ssub[ds][0])):
                    cor[mapIdx[n], i] = np.nan_to_num(
                        np.corrcoef(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                                    comp[mapIdx[n]])[0, 1])
        return cor

    for i, ssub in enumerate(ssubs):
        cor = foo(ssub, compare)
        plt.plot(dsls, np.mean(cor, 0), ls=ls, lw=4, c=colors[i],
                 label=None if labels is None else labels[i], clip_on=False)
        noclip(plt.errorbar(dsls, np.mean(cor, 0), yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                            lw=3, capthick=2, fmt='o', c=colors[i], clip_on=False))
    plt.xlabel('Spatial decimation')
    simpleaxis(plt.gca())
    plt.xticks(dsls, ['1', '', '', '', '', '8x8',
                      '', '16x16', '24x24', '32x32'])
    plt.yticks(*[[.4, .6, .8, 1.0]] * 2)
    plt.xlim(dsls[0], dsls[-1])
    plt.ylabel('Correlation w/ undecimated $C_1$/$S_1$', y=.42, labelpad=1)
    if labels is not None:
        lg = plt.legend(loc=(0, 0))
        return lg


ssub = np.load('results/decimate.npz')['ssub'].item()
il2 = np.load('results/decimate-interleave.npz')['il2'].item()

plt.figure()
first_legend = plotCorr([ssub, il2], trueC, dsls, ca_or_spikes='ca',
                        labels=['no interleaving', 'interleaving'], colors=[cyan, green])
plotCorr([ssub, il2], trueSpikes, dsls, ca_or_spikes='spikes', ls='--', colors=[cyan, green])
l1, = plt.plot([0, 1], [-1, -1], lw=4, c='k', label='denoised')
l2, = plt.plot([0, 1], [-1, -1], lw=4, c='k', ls='--', label='deconvolved')
# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)
# Create another legend for the second line.
plt.legend(handles=[l1, l2], loc=(0, .21))
plt.ylim(.36, 1)
plt.subplots_adjust(.13, .15, .94, .96)
plt.savefig(figpath + '/Corr-interleave.pdf') if figpath else showpause()

#

# artificial data, from 'true' generative model by shuffling residuals in time

ssub = np.load('results/decimate-stratshuffled.npz')['ssub'].item()
il2 = np.load('results/decimate-interleave-stratshuffled.npz')['il2'].item()

plt.figure()
plotCorr([ssub, il2], trueC, dsls, ca_or_spikes='ca', colors=[cyan, green])
plotCorr([ssub, il2], trueSpikes, dsls, ca_or_spikes='spikes', ls='--', colors=[cyan, green])
l1, = plt.plot([0, 1], [-1, -1], lw=4, c='k', label='denoised')
l2, = plt.plot([0, 1], [-1, -1], lw=4, c='k', ls='--', label='deconvolved')
plt.ylabel(r'Correlation w/ ground truth $C$\textsuperscript{s}/$S$\textsuperscript{s}',
           y=.42, labelpad=3)
plt.ylim(.36, 1)
plt.subplots_adjust(.13, .15, .94, .96)
plt.savefig(figpath + '/Corr-interleave-stratshuffled.pdf') if figpath else plt.show(block=True)
