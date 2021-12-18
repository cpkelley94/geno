from cellpose import models
from matplotlib import pyplot as plt
from skimage import filters, morphology
from xml.etree import ElementTree
import argparse
import contextlib
import csv
import io
import mxnet as mx
import numpy as np
import os
import sys

# custom libraries
import scope_utils3 as su

# print colors
TGREEN = '\033[32m'
TRED = '\033[31m'
TRETURN = '\033[m'


#--  FUNCTION DECLARATIONS  ---------------------------------------------------#

@contextlib.contextmanager
def silence_stdout():
    '''
    Prevent a block of code from printing to the terminal. To use, open the \
    context with the `with` keyword.
    '''
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = save_stdout


#--  INITIALIZATION AND COMMAND LINE INTERFACE  -------------------------------#

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('czi', type=str, nargs=1, help='Path to image file.')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('-m', '--meta', help='Path to XML image metadata (required if using OME-TIFF).', default=None)

# parse arguments
args = vars(parser.parse_args())
img_path = args['czi'][0]
outdir = args['outdir'][0]
meta_path = args['meta']

if not os.path.exists(outdir):
    os.makedirs(outdir)

# initialize cellpose for nucleus and cell segmentation
print('\nInitializing cellpose...', end='')
with silence_stdout():
    dev = mx.cpu()
    model_nuc = models.Cellpose(dev, model_type='nuclei')
    model_cyt = models.Cellpose(dev, model_type='cyto')
print('Done.')


#--  FILE OPERATIONS  ---------------------------------------------------------#

# determine image filetype and open
print('Loading image...', end='')

if img_path.lower().endswith('.czi'):
    img_type = 'czi'
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # open CZI format using `czifile` library
    from czifile import CziFile
    with CziFile(img_path) as czi_file:
        img_czi = czi_file.asarray()
        if meta_path is not None:
            mtree = ElementTree.parse(meta_path)
        else:
            meta = czi_file.metadata()
            mtree = ElementTree.fromstring(meta)

    img = img_czi[0,0,:,0,0,:,:,0]  # c, z, x, y

else:
    raise ValueError('Image filetype not recognized. Allowed:  .CZI')

print('Done.')

# get pixel dimensions from CZI metadata
dim_iter = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
dims = {}
for dimension in dim_iter:
    dim_id = dimension.get('Id')
    dims.update({dim_id:float(dimension.find('Value').text)*1.E6}) # [um]
pixel_area = dims['X'] * dims['Y']  # [um^2]

# get channel filters from CZI metadata
track_tree = mtree.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup').findall('TrackSetup')
tracks = []
for track in track_tree:
    filtr = track.find('BeamSplitters').find('BeamSplitter').find('Filter').text
    tracks.append(filtr)
print('\nTracks: ' + ', '.join(tracks))

# split channels into 2D arrays
img_mbnl = su.normalize_image(img[0,:,:])
img_dcas13 = su.normalize_image(img[1,:,:])
img_fish = su.normalize_image(img[2,:,:])
img_dapi = su.normalize_image(img[3,:,:])

# draw channels
img_list = [img_mbnl, img_dcas13, img_fish, img_dapi]
titles = ['mbnl1', 'dcas13d', 'cug fish', 'dapi']
vmaxs = [0.5, 0.5, 0.75, 1]
cmaps = [su.cmap_KtoB, su.cmap_KtoG, su.cmap_KtoR, 'binary_r']
fig, ax = plt.subplots(1, len(img_list), figsize=(3.*len(img_list), 3.))
imxy = []
for i, img in enumerate(img_list):
    imxy.append(ax[i].imshow(img, vmax=vmaxs[i], cmap=cmaps[i]))
    ax[i].set_title(titles[i])
plt.tight_layout()
plt.savefig(os.path.join(outdir, img_name + '_channels.pdf'), dpi=300)
plt.close()
    

#--  PROCESSING  --------------------------------------------------------------#

# segment nuclei
print('\nSegmenting nuclei...', end='')
with silence_stdout():
    masks_nuc, _, _, _ = model_nuc.eval([img_dapi], rescale=None, channels=[[0, 0]], diameter=80.)
    labeled_mask_nuc = masks_nuc[0]
    labeled_mask_nuc = morphology.remove_small_objects(labeled_mask_nuc, 256)
nuclei_labels = sorted(list(set(list(labeled_mask_nuc.flatten()))))[1:]
nuclei_coords = {l:[tuple(row) for row in np.argwhere(labeled_mask_nuc == l)] for l in nuclei_labels}
print('Done.')

# identify FISH-positive cells
bg_intens = np.median(img_fish[labeled_mask_nuc == 0])
positive_labels = []
positive_mask = np.zeros_like(labeled_mask_nuc)
for l in nuclei_labels:
    max98_intens = np.percentile(img_fish[labeled_mask_nuc == l], 98)
    print(max98_intens/bg_intens)
    if max98_intens/bg_intens > 25:  # positive
        positive_labels.append(l)
        positive_mask = positive_mask + (labeled_mask_nuc == l).astype(int)

negative_mask = (labeled_mask_nuc > 0).astype(int) - positive_mask

# output segmentation
fig, ax = plt.subplots(1, 3, figsize=(10,4))
ax[0].imshow(img_dapi, cmap=su.cmap_KtoB)
ax[1].imshow(img_fish, cmap=su.cmap_KtoR)
ax[2].imshow(negative_mask, cmap='binary_r')
ax[2].imshow(positive_mask, cmap=su.cmap_NtoR)
plt.savefig(os.path.join(outdir, img_name + '_mask.png'), dpi=300)
plt.close()

# segment foci for each positive nucleus
img_fish_blur = filters.gaussian(img_fish, (1, 1))
output_list = []
for l in positive_labels:
    this_nuc = (labeled_mask_nuc == l)
    for i in range(15):
        this_nuc = morphology.binary_erosion(this_nuc.astype(int)).astype(bool)
    seg_foci = np.logical_and(this_nuc, img_fish_blur > 5*np.median(img_fish[this_nuc]))
    seg_not_foci = np.logical_and(this_nuc, np.logical_not(seg_foci))

    # output segmentation
    fig, ax = plt.subplots(1, 4, figsize=(10,4))
    ax[0].imshow(this_nuc, cmap=su.cmap_KtoB)
    ax[1].imshow(img_fish, cmap=su.cmap_KtoR)
    ax[2].imshow(seg_foci, cmap='binary_r')
    ax[3].imshow(seg_not_foci, cmap='binary_r')
    plt.savefig(os.path.join(outdir, img_name + '_fish_seg_cell' + str(l) + '.png'), dpi=300)
    plt.close()

    # calculate enrichments
    enrich_dcas13 = np.mean(img_dcas13[seg_foci])/np.mean(img_dcas13[seg_not_foci])
    enrich_mbnl1 = np.mean(img_mbnl[seg_foci])/np.mean(img_mbnl[seg_not_foci])

    output_line = [l, enrich_dcas13, enrich_mbnl1]
    output_list.append(output_line)

# output enrichments
with open(os.path.join(outdir, img_name + '_enrichments.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(output_list)

