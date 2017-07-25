import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter
from functions import init_fig, simpleaxis, showpause, IQRfill
import ca_source_extraction as cse  # https://github.com/j-friedrich/CaImAn/tree/multi-scale_paper
from scipy.stats import pearsonr

if matplotlib.__version__[0] == '2':
    matplotlib.style.use('classic')

try:
    from sys import argv
    from os.path import isdir
    figpath = argv[1] if isdir(argv[1]) else False
except:
    figpath = False

matplotlib.style.use('classic')
init_fig()

# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
[vermillon, orange, yellow, green, cyan, blue, purple, grey] = col

#

# Load results

dsls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

f, A2, b2, C2 = itemgetter('f', 'A2', 'b2', 'C2')(np.load('results/CNMF-HRshapes.npz'))
A2 = A2.item()
N = A2.shape[1]

res = np.load('results/decimate.npz')
ssub, ssubX = res['ssub'].item(), res['ssubX'].item()
srt = cse.utilities.order_components(A2, ssub[1][0])[-1]
trueC = ssub[1][0]
trueSpikes = ssub[1][2]

res = np.load('results/decimate-LR.npz')
ssub0, ssubX0 = res['ssub'].item(), res['ssubX'].item()


#

# # plot correlations

def plotCorr(ssub, ssub0, r=pearsonr, ds1phase=[1, 2, 3, 4], loc=(.1, .01)):
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
    plt.ylim(.3, 1)
    plt.yticks(
        *[np.round(np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.2), 1)] * 2)
    plt.xlim(dsls[0], dsls[-1])
    plt.ylabel('Correlation w/ undecimated $C_1$/$S_1$', y=.42, labelpad=1)
    plt.legend(handles=[l3, l4, l1, l2], loc=loc, ncol=1)
    plt.subplots_adjust(.13, .15, .94, .96)
    return l1, l2, l3, l4


plt.figure()
plotCorr(ssub, ssub0)
plt.ylim(.25, 1)
plt.savefig(figpath + '/Corr.pdf', transparent=True) if figpath else showpause()

plt.figure()
plotCorr(ssubX, ssubX0)
plt.ylim(.4, 1)
plt.xticks(dsls, ['1', '', '', '', '', '8x1', '', '16x1', '24x1', '32x1'])
plt.savefig(figpath + '/xCorr.pdf', transparent=True) if figpath else showpause()

#

# # artificial data, from 'true' generative model by shuffling residuals in time

res = np.load('results/decimate-stratshuffled.npz')
ssub, ssubX = res['ssub'].item(), res['ssubX'].item()
res = np.load('results/decimate-stratshuffled-LR.npz')
ssub0, ssubX0 = res['ssub'].item(), res['ssubX'].item()


plt.figure()
plotCorr(ssub, ssub0)
plt.ylabel(r'Correlation w/ ground truth $C$\textsuperscript{s}/$S$\textsuperscript{s}',
           y=.42, labelpad=3)
plt.ylim(.35, 1)
plt.savefig(figpath + '/Corr-stratshuffled.pdf', transparent=True) if figpath else showpause()


plt.figure()
plotCorr(ssubX, ssubX0)
plt.ylabel(r'Correlation w/ ground truth $C$\textsuperscript{s}/$S$\textsuperscript{s}',
           y=.42, labelpad=3)
plt.ylim(.2, 1)
plt.xticks(dsls, ['1', '', '', '', '', '8x1', '', '16x1', '24x1', '32x1'])
plt.savefig(figpath + '/xCorr-stratshuffled.pdf', transparent=True)\
    if figpath else plt.show(block=True)
