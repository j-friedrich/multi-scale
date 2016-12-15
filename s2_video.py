# writes every frame to a file and uses ffmpeg to asseble the video
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import ca_source_extraction as cse
import tifffile
from os import system
from skimage.filters import gaussian
from scipy.sparse import coo_matrix

filename = '180um_20fps_350umX350um.tif'
data = tifffile.TiffFile(filename).asarray()
T, X, Y = data.shape

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


ds = 16
plot_smooth_shapes = True

system("mkdir tmp4video")
res = np.load('results/CNMF-HRshapes.npz')
A2 = res['A2'].item()
b2 = res['b2']
del res

ssub1 = np.load('results/decimate.npz')['ssub'].item()[1]
denoised1 = b2.dot(ssub1[1]).astype('float32') + A2.dot(ssub1[0]).astype('float32')
denoised1.shape = (X, Y, T)
denoised1 = np.transpose(denoised1, (2, 1, 0))
residual1 = data - denoised1
mi = np.percentile(data, 10)
ma = np.percentile(data, 99.5)  # 99.8
maR = max(np.percentile(residual1, 98), -np.percentile(residual1, 2))

ssub = np.load('results/decimate.npz')['ssub'].item()[ds]
denoised = b2.dot(ssub[1]).astype('float32') + A2.dot(ssub[0]).astype('float32')
denoised.shape = (X, Y, T)
denoised = np.transpose(denoised, (2, 1, 0))
residual = data - denoised
if plot_smooth_shapes:
    A2 = coo_matrix(np.transpose([gaussian(a.reshape(X, Y), 1).ravel()
                                  for a in A2.toarray().T]))

fig = plt.figure(figsize=(16.7, 10))
ax1 = fig.add_axes([.002, .512, .3, .46])

cse.plot_contours(A2, data[0], thr=0.9, display_numbers=False, colors='w')
im1 = plt.imshow(data[0], cmap=gfp, vmin=mi, vmax=ma)
plt.title('data')
plt.axis('off')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax2 = fig.add_axes([.332, .512, .3, .46])
cse.plot_contours(A2, denoised1[0], thr=0.9, display_numbers=False, colors='w')
im2 = plt.imshow(denoised1[0], cmap=gfp, vmin=mi, vmax=ma)
plt.title('denoised')
plt.axis('off')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax3 = fig.add_axes([.662, .512, .3, .46])
cse.plot_contours(A2, residual1[0], thr=0.9, display_numbers=False, colors='w')
im3 = plt.imshow(residual1[0], cmap=blue2green, vmin=-maR, vmax=maR)
plt.title('residual')
plt.axis('off')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

ax4 = fig.add_axes([.002, .005, .3, .46])
cse.plot_contours(A2, data[0], thr=0.9, display_numbers=False, colors='w')
im4 = plt.imshow(data[0].reshape((X / ds, ds, Y / ds, ds)).mean(-1).mean(-2)
                 .repeat(ds, 0).repeat(ds, 1), cmap=gfp, vmin=mi, vmax=ma)
plt.title('low resolution data (downsampled by %dx%d)' % (ds, ds))
plt.axis('off')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax5 = fig.add_axes([.332, .005, .3, .46])
cse.plot_contours(A2, denoised[0], thr=0.9, display_numbers=False, colors='w')
im5 = plt.imshow(denoised[0], cmap=gfp, vmin=mi, vmax=ma)
plt.title('high resolution reconstruction')
plt.axis('off')
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
ax6 = fig.add_axes([.662, .005, .3, .46])
cse.plot_contours(A2, residual[0], thr=0.9, display_numbers=False, colors='w')
im6 = plt.imshow(residual[0], cmap=blue2green, vmin=-maR, vmax=maR)
plt.title('high resolution residual')
plt.axis('off')
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

for i in range(len(data)):
    im1.set_data(data[i])
    im2.set_data(denoised1[i])
    im3.set_data(residual1[i])
    im4.set_data(data[i].reshape((X / ds, ds, Y / ds, ds)).mean(-1).mean(-2)
                 .repeat(ds, 0).repeat(ds, 1))
    im5.set_data(denoised[i])
    im6.set_data(residual[i])
    plt.savefig("tmp4video/%04d.png" % i)
# High Efficiency Video Coding (HEVC), also known as H.265
system("ffmpeg -r 60 -i tmp4video/%04d.png -c:v libx265 -r 60 -preset veryslow video_2P_x265.mp4")
# H.264/AVC for greater compatibility, e.g. quicktime
system("ffmpeg -r 60 -i tmp4video/%04d.png -c:v libx264 -r 60 -preset veryslow -pix_fmt yuv420p -crf 28 video_2P.mp4")
# system("rm -r tmp4video")
