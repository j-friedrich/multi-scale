import numpy as np
import ca_source_extraction as cse
import matplotlib
import matplotlib.pyplot as plt
import tifffile
from operator import itemgetter
from skimage.filters import gaussian
from functions import init_fig


init_fig()
plt.rc('font', size=20, **{'family': 'sans-serif',
                           'sans-serif': ['Computer Modern']})
save_figs = False  # subfolder fig must exist if set to True


cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
gfp = matplotlib.colors.LinearSegmentedColormap('GFP_colormap', cdict, 256)

#

# Load data

d1, d2, T = (512, 512, 2000)
filename = '180um_20fps_350umX350um.tif'
Yr = tifffile.TiffFile(filename).asarray().astype(dtype='float32')
Yr = np.transpose(Yr, (1, 2, 0))
Ymax = Yr.max(-1)
Yr = np.reshape(Yr, (d1 * d2, T), order='F')

A2, b2 = itemgetter('A2', 'b2')(np.load('results/CNMF-HRshapes.npz'))
A2 = A2.item()
N = A2.shape[1]

ssub = np.load('results/decimate.npz')['ssub'].item()

A_or, C_or, srt = cse.utilities.order_components(A2, ssub[1][0])
A_or = np.transpose([gaussian(a.reshape(d1, d2), 1).ravel() for a in A_or.T])

#

# Plot data

fig = plt.figure(figsize=(13, 12))
fig.add_axes([0, .5, .46, .46])
crd = cse.utilities.plot_contours(
    A_or, Ymax, thr=0.9, display_numbers=False, colors='w')
ax = plt.gca()
ax.imshow(Yr.reshape(-1, T / 20, 20).mean(-1).max(-1).reshape(d1, d2).T,
          cmap=gfp, vmin=400, vmax=4000)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
plt.title('data', fontsize=20)

fig.add_axes([.45, .5, .46, .46])
crd = cse.utilities.plot_contours(
    A_or, Ymax, thr=0.9, display_numbers=False, colors='w')
ax = plt.gca()
denoised = b2.dot(ssub[1][1]).astype('float32') + \
    A2.dot(ssub[1][0]).astype('float32')
ax.imshow(denoised.reshape(d1, d2, T / 10, 10).mean(-1).max(-1).T,
          cmap=gfp, vmin=400, vmax=4000)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
plt.title('denoised', fontsize=20)

fig.add_axes([0, .005, .46, .46])
crd = cse.utilities.plot_contours(
    A_or, Ymax, thr=0.9, display_numbers=False, colors='w')
ax = plt.gca()
ax.imshow(Yr.reshape(d1 / 16, 16, d2 / 16, 16, T / 10, 10).mean(-1).mean(-2).mean(-3)
          .max(-1).T.repeat(16, 0).repeat(16, 1), cmap=gfp, vmin=400, vmax=4000)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
plt.title('low resolution data (downsampled by 16x16)', fontsize=20)

fig.add_axes([.45, .005, .46, .46])
crd = cse.utilities.plot_contours(
    A_or, Ymax, thr=0.9, display_numbers=False, colors='w')
ax = plt.gca()
denoised = b2.dot(ssub[16][1]).astype('float32') + \
    A2.dot(ssub[16][0]).astype('float32')
im = ax.imshow(denoised.reshape(d1, d2, T / 10, 10).mean(-1).max(-1).T,
               cmap=gfp, vmin=400, vmax=4000)
del denoised  # free memory
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
plt.title('high resolution reconstruction', fontsize=20)

cax = fig.add_axes([.91, .008, .02, .95])
cb = plt.colorbar(im, cax=cax, label='Maximum Projection')
cb.set_ticks(range(0, 4000, 1000))
plt.yticks(range(0, 4000, 1000), range(1000, 5000, 1000), fontsize=18)
cb.ax.yaxis.label.set_font_properties(
    matplotlib.font_manager.FontProperties(size=18))

if save_figs:
    plt.savefig('fig/data+reconstruction.pdf', bbox_inches='tight', pad_inches=.01)
plt.show()
