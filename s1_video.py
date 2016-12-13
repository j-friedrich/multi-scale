# writes every frame to a file and uses ffmpeg to asseble the video
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os import system
from skimage.filters import gaussian
from scipy.sparse import coo_matrix
from CNMF import LocalNMF, HALS4activity
import ca_source_extraction as cse  # github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper


# define colormaps
cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
gfp = matplotlib.colors.LinearSegmentedColormap('GFP_colormap', cdict, 256)
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.5, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
blue2green = matplotlib.colors.LinearSegmentedColormap('Blue2Green_colormap', cdict, 256)

data = np.load('data_zebrafish.npy')
T, X, Y = data.shape
sig = (3, 3)
ds = 8
plot_smooth_shapes = True

# find neurons greedily
N = 43
centers = cse.greedyROI(data.reshape((-1, 30) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[4, 4])[2]

MSE_array, shapes, activity, boxes = LocalNMF(data, centers, sig, iters=40, mb=20, iters0=30)
# scale
z = np.sqrt(np.sum(shapes.reshape(len(shapes), -1)[:-1]**2, 1)).reshape(-1, 1)
activity[:-1] *= z
z += (z == 0)  # avoids divide by zero in next line, i.e. keep shape=0
shapes[:-1] /= z.reshape(-1, 1, 1)
# reconstruction and residual based on high-res data
denoised1 = activity.T.dot(shapes.reshape(N + 1, -1)).reshape(data.shape)
residual1 = data - denoised1

# plot range
mi = 101
ma = 290
maR = 55

# acticity inferred from decimated data
activity_ds = HALS4activity(data.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                            .mean(-1).mean(-2).reshape(len(data), -1),
                            shapes.reshape(-1, 96 / ds, ds, 96 / ds, ds)
                            .mean(-1).mean(-2).reshape(len(shapes), -1),
                            activity.copy(), 20)
# reconstruction and residual based on low-res data
denoised = activity_ds.T.dot(shapes.reshape(N + 1, -1)).reshape(data.shape)
residual = data - denoised

if plot_smooth_shapes:
    A2 = coo_matrix(np.transpose([gaussian(a, 1).ravel() for a in shapes[:N]]))
else:
    A2 = coo_matrix(shapes[:N].reshape(N, -1).T)

fig = plt.figure(figsize=(16.7, 10))
ax1 = fig.add_axes([.002, .512, .3, .46])
cse.plot_contours(A2, data[0].T, thr=0.9, display_numbers=False, colors='w')
im1 = plt.imshow(data[0].T, cmap=gfp, vmin=mi, vmax=ma)
plt.title('data')
plt.axis('off')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax2 = fig.add_axes([.332, .512, .3, .46])
cse.plot_contours(A2, denoised1[0].T, thr=0.9, display_numbers=False, colors='w')
im2 = plt.imshow(denoised1[0].T, cmap=gfp, vmin=mi, vmax=ma)
plt.title('denoised')
plt.axis('off')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax3 = fig.add_axes([.662, .512, .3, .46])
cse.plot_contours(A2, residual1[0].T, thr=0.9, display_numbers=False, colors='w')
im3 = plt.imshow(residual1[0].T, cmap=blue2green, vmin=-maR, vmax=maR)
plt.title('residual')
plt.axis('off')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

ax4 = fig.add_axes([.002, .005, .3, .46])
cse.plot_contours(A2, data[0].T, thr=0.9, display_numbers=False, colors='w')
im4 = plt.imshow(data[0].reshape((X / ds, ds, Y / ds, ds)).mean(-1).mean(-2)
                 .repeat(ds, 0).repeat(ds, 1).T, cmap=gfp, vmin=mi, vmax=ma)
plt.title('low resolution data (downsampled by %dx%d)' % (ds, ds))
plt.axis('off')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax5 = fig.add_axes([.332, .005, .3, .46])
cse.plot_contours(A2, denoised[0].T, thr=0.9, display_numbers=False, colors='w')
im5 = plt.imshow(denoised[0].T, cmap=gfp, vmin=mi, vmax=ma)
plt.title('high resolution reconstruction')
plt.axis('off')
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax6 = fig.add_axes([.662, .005, .3, .46])
cse.plot_contours(A2, residual[0].T, thr=0.9, display_numbers=False, colors='w')
im6 = plt.imshow(residual[0].T, cmap=blue2green, vmin=-maR, vmax=maR)
plt.title('high resolution residual')
plt.axis('off')
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

system("mkdir tmp4video")
for i in range(len(data)):
    im1.set_data(data[i].T)
    im2.set_data(denoised1[i].T)
    im3.set_data(residual1[i].T)
    im4.set_data(data[i].reshape((X / ds, ds, Y / ds, ds)).mean(-1).mean(-2)
                 .repeat(ds, 0).repeat(ds, 1).T)
    im5.set_data(denoised[i].T)
    im6.set_data(residual[i].T)
    plt.savefig("tmp4video/%04d.png" % i)

# High Efficiency Video Coding (HEVC), also known as H.265
system("ffmpeg -r 60 -i tmp4video/%04d.png -c:v libx265 -r 60 -preset veryslow video_zebrafish_x265.mp4")
# H.264/AVC for greater compatibility, e.g. quicktime
system("ffmpeg -r 60 -i tmp4video/%04d.png -c:v libx264 -r 60 -preset veryslow -pix_fmt yuv420p -crf 28 video_zebrafish.mp4")
# system("rm -r tmp4video")
