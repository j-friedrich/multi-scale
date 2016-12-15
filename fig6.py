import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from operator import itemgetter
from functions import init_fig, simpleaxis, noclip
import ca_source_extraction as cse


init_fig()
save_figs = False  # subfolder fig must exist if set to True


# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
[vermillon, orange, yellow, green, cyan, blue, purple, grey] = col

#

# Load data

dsls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

f, A2, b2, C2 = itemgetter('f', 'A2', 'b2', 'C2')(
    np.load('results/CNMF-HRshapes.npz'))
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


def plotCorr(ssub, ssub0):
    def foo(ssub, comp, dsls=dsls, ca_or_spikes='ca'):
        N, T = comp.shape
        cor = np.zeros((N, len(dsls))) * np.nan
        for i, ds in enumerate(dsls):
            if len(ssub[ds][0]) == len(comp):
                cor[:, i] = np.array(
                    [np.corrcoef(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                                 comp[n])[0, 1] for n in range(N)])
            else:  # necessary if update_spatial_components removed a component
                mapIdx = [np.argmax([np.corrcoef(s, tC)[0, 1] for tC in comp])
                          for s in ssub[ds][0 if ca_or_spikes == 'ca' else 2]]
                for n in range(len(ssub[ds][0])):
                    cor[mapIdx[n], i] = np.array(
                        np.corrcoef(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                                    comp[mapIdx[n]])[0, 1])
        return np.nan_to_num(cor)

    cor = foo(ssub0, trueC, [1, 2, 3, 4])
    l1, = plt.plot([1, 2, 3, 4], np.nanmean(cor, 0), lw=4, c=cyan,
                   label='1 phase imaging', clip_on=False)
    noclip(plt.errorbar([1, 2, 3, 4], np.mean(cor, 0), yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                        lw=3, capthick=2, fmt='o', c=cyan, clip_on=False))

    cor = foo(ssub, trueC)
    l2, = plt.plot(dsls, np.mean(cor, 0), lw=4, c=orange,
                   label='2 phase imaging', clip_on=False)
    noclip(plt.errorbar(dsls, np.mean(cor, 0), yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                        lw=3, capthick=2, fmt='o', c=orange, clip_on=False))

    cor = foo(ssub0, trueSpikes, [1, 2, 3, 4], ca_or_spikes='spikes')
    plt.plot([1, 2, 3, 4], np.mean(cor, 0), lw=4,
             c=cyan, ls='--', clip_on=False)
    noclip(plt.errorbar([1, 2, 3, 4], np.mean(cor, 0), yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                        lw=3, capthick=2, fmt='o', c=cyan, clip_on=False))

    cor = foo(ssub, trueSpikes, ca_or_spikes='spikes')
    plt.plot(dsls, np.mean(cor, 0), lw=4, c=orange, ls='--', clip_on=False)
    noclip(plt.errorbar(dsls, np.mean(cor, 0), yerr=np.std(cor, 0) / np.sqrt(len(cor)),
                        lw=3, capthick=2, fmt='o', c=orange, clip_on=False))

    l3, = plt.plot([0, 1], [-1, -1], lw=4, c='k', label='denoised')
    l4, = plt.plot([0, 1], [-1, -1], lw=4, c='k', ls='--', label='deconvolved')

    plt.xlabel('Spatial decimation')
    simpleaxis(plt.gca())
    plt.xticks(dsls, ['1', '', '', '', '', '8x8',
                      '', '16x16', '24x24', '32x32'])
    plt.yticks(
        *[np.round(np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.2), 1)] * 2)
    plt.xlim(dsls[0], dsls[-1])
    plt.ylim(.3, 1)
    plt.ylabel('Correlation w/ undecimated $C_1$/$S_1$', y=.42, labelpad=1)
    plt.legend(handles=[l3, l4, l1, l2], loc=(.15, .01), ncol=1)
    plt.subplots_adjust(.13, .15, .94, .96)
    return l1, l2, l3, l4


plt.figure()
plotCorr(ssub, ssub0)
if save_figs:
    plt.savefig('fig/Corr.pdf', transparent=True)
plt.show()

plt.figure()
plotCorr(ssubX, ssubX0)
plt.ylim(.45, 1)
plt.xticks(dsls, ['1', '', '', '', '', '8x1', '', '16x1', '24x1', '32x1'])
if save_figs:
    plt.savefig('fig/xCorr.pdf', transparent=True)
plt.show()


def foo(ssub, comp, dsls=dsls, ca_or_spikes='ca'):
    N, T = comp.shape
    cor = np.zeros((N, len(dsls))) * np.nan
    for i, ds in enumerate(dsls):
        cor[:, i] = np.array(
            [np.corrcoef(ssub[ds][0 if ca_or_spikes == 'ca' else 2][n],
                         comp[n])[0, 1] for n in range(N)])
    return np.nan_to_num(cor)


cor = foo(ssub, trueC)
fig = plt.figure()
for i, n in enumerate(srt[::-1]):
    plt.plot(dsls, cor[n], lw=1, c=plt.cm.rainbow(i / float(N)))
plt.xlabel('Spatial decimation')
simpleaxis(plt.gca())
plt.xticks(dsls, ['1', '', '', '', '', '8x8', '', '16x16', '24x24', '32x32'])
plt.yticks(
    *[np.round(np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.2), 1)] * 2)
plt.xlim(dsls[0], dsls[-1])
plt.ylim(.1, 1)
plt.ylabel('Correlation with undecimated $C_1$', y=.45, labelpad=1)
# create a second axes for the colorbar
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.15)
bounds = range(1, N + 1)
cb = matplotlib.colorbar.ColorbarBase(
    cax, cmap='rainbow', spacing='proportional', boundaries=bounds)
cb.set_ticks([N, 1])
plt.yticks([N, 1], [1, N])
cb.ax.yaxis.label.set_font_properties(
    matplotlib.font_manager.FontProperties(size=20))
cax.set_ylabel('Rank', size=30, labelpad=-30)
plt.subplots_adjust(.13, .15, .92, .96)
if save_figs:
    plt.savefig('fig/Corr-indiv.pdf', transparent=True)
plt.show()

fig = plt.figure()
for i, n in enumerate(srt[::-1]):
    plt.plot(dsls[1:], -np.log(1 - cor[n, 1:]),
             lw=1, c=plt.cm.rainbow(i / float(N)))
plt.xlabel('Spatial decimation')
simpleaxis(plt.gca())
plt.xticks(dsls, ['1', '', '', '', '', '8x8', '', '16x16', '24x24', '32x32'])
plt.yticks(-np.log(1 -
                   np.array([0, .9, .99, .999, .9999, .99999])), [0, '', .99, '', .9999, ''])
plt.xlim(dsls[0], dsls[-1])
plt.ylabel('Correlation', labelpad=-20)
# create a second axes for the colorbar
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.15)
bounds = range(1, N + 1)
cb = matplotlib.colorbar.ColorbarBase(
    cax, cmap='rainbow', spacing='proportional', boundaries=bounds)
cb.set_ticks([N, 1])
plt.yticks([N, 1], [1, N])
cb.ax.yaxis.label.set_font_properties(
    matplotlib.font_manager.FontProperties(size=20))
cax.set_ylabel('Rank', size=30, labelpad=-30)
plt.subplots_adjust(.16, .15, .92, .96)
if save_figs:
    plt.savefig('fig/Corr-indiv_log.pdf')
plt.show()

#

# # artificial data, from 'true' generative model by shuffling residuals in time

res = np.load('results/decimate-shuffled.npz')
ssub, ssubX = res['ssub'].item(), res['ssubX'].item()
res = np.load('results/decimate-shuffled-LR.npz')
ssub0, ssubX0 = res['ssub'].item(), res['ssubX'].item()


plt.figure()
plotCorr(ssub, ssub0)
plt.ylabel(r'Correlation w/ ground truth $C$\textsuperscript{s}/$S$\textsuperscript{s}',
           y=.42, labelpad=3)
if save_figs:
    plt.savefig('fig/Corr-shuffled.pdf', transparent=True)
plt.show()

plt.figure()
plotCorr(ssubX, ssubX0)
plt.ylabel(r'Correlation w/ ground truth $C$\textsuperscript{s}/$S$\textsuperscript{s}',
           y=.42, labelpad=3)
plt.ylim(.25, 1)
plt.xticks(dsls, ['1', '', '', '', '', '8x1', '', '16x1', '24x1', '32x1'])
if save_figs:
    plt.savefig('fig/xCorr-shuffled.pdf', transparent=True)
plt.show()


cor = foo(ssub, trueC)
fig = plt.figure()
for i, n in enumerate(srt[::-1]):
    plt.plot(dsls, cor[n], lw=1, c=plt.cm.rainbow(i / float(N)))
plt.xlabel('Spatial decimation')
simpleaxis(plt.gca())
plt.xticks(dsls, ['1', '', '', '', '', '8x8', '', '16x16', '24x24', '32x32'])
plt.yticks(
    *[np.round(np.arange(np.round(plt.ylim()[1], 1), plt.ylim()[0], -.2), 1)] * 2)
plt.xlim(dsls[0], dsls[-1])
plt.ylim(.1, 1)
plt.ylabel(r'Correlation w/ ground truth $C$\textsuperscript{s}', y=.45, labelpad=5)
# create a second axes for the colorbar
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.15)
bounds = range(1, N + 1)
cb = matplotlib.colorbar.ColorbarBase(
    cax, cmap='rainbow', spacing='proportional', boundaries=bounds)
cb.set_ticks([N, 1])
plt.yticks([N, 1], [1, N])
cb.ax.yaxis.label.set_font_properties(
    matplotlib.font_manager.FontProperties(size=20))
cax.set_ylabel('Rank', size=30, labelpad=-30)
plt.subplots_adjust(.13, .15, .92, .96)
if save_figs:
    plt.savefig('fig/Corr-indiv-shuffled.pdf', transparent=True)
plt.show()
