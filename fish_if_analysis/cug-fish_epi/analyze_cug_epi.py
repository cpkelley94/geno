# detect environment
import os
hostname = os.popen('hostname').read()
if 'ufhpc' in hostname:
    # hipergator
    import matplotlib
    matplotlib.use('Agg')
    env = 'ufhpc'
else:
    env = 'local'

# imports
from cellpose import models
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import feature, filters, morphology
from xml.etree import ElementTree
import argparse
import contextlib
import csv
import hashlib
import io
import mxnet as mx
import numpy as np
# import os
import scipy.stats as ss
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

    img = img_czi[0,0,:,0,:,:,:,0]  # c, z, x, y

elif any([img_path.lower().endswith(ext) for ext in ['.ome.tiff', '.ome.tif']]):
    img_type = 'ome-tiff'
    img_name = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]

    # open OME-TIFF format using `bioformats` library (requires Java)
    import javabridge
    import bioformats

    with silence_stdout():
        # read file with Java BioFormats
        javabridge.start_vm(class_path=bioformats.JARS)
        s = bioformats.load_image(img_path, z=0)
        javabridge.kill_vm()

    img_ome = np.array(s)
    img = img_ome.transpose(2,0,1)  # c, z, x, y

    # look for metadata .XML file with same filename
    if meta_path is None:
        meta_path = os.path.splitext(os.path.splitext(img_path)[0])[0] + '.xml'
    try:
        mtree = ElementTree.parse(meta_path)
    except IOError:
        # metadata file not found
        raise IOError('CZI metadata XML not found at expected path "' + meta_path + '" (required for OME-TIFF)')

else:
    raise ValueError('Image filetype not recognized. Allowed:  .CZI, .OME.TIFF')

print('Done.')

# get pixel dimensions
dim_iter = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
dims = {}
for dimension in dim_iter:
    dim_id = dimension.get('Id')
    dims.update({dim_id:float(dimension.find('Value').text)*1.E6}) # [um]
pixel_area = dims['X'] * dims['Y']  # [um^2]

# get channel filters
track_tree = mtree.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup').findall('TrackSetup')
tracks = []
for track in track_tree:
    filtr = track.find('BeamSplitters').find('BeamSplitter').find('Filter').text
    tracks.append(filtr)
print('\nTracks: ' + ', '.join(tracks))

# split channels
try:
    img_dapi = su.normalize_image(img[tracks.index('FSet49 wf'),:,:,:])
    print('DAPI: ' + str('FSet49 wf'))
except ValueError:
    # use first channel
    img_dapi = su.normalize_image(img[0,:,:,:])
    print('DAPI: ' + str(tracks[0]))

try:
    img_gfp = su.normalize_image(img[tracks.index('FSet38 HE'),:,:,:])
    print('GFP: ' + str('FSet38 HE'))
except ValueError:
    # use next channel
    img_gfp = su.normalize_image(img[1,:,:,:])
    print('GFP: ' + str(tracks[1]))

try:
    img_fish = img[tracks.index('FSet45 wf'),:,:,:]
    print('FISH: ' + str('FSet49 wf'))
except ValueError:
    # use last channel
    img_fish = img[-1,:,:,:]
    print('GFP: ' + str(tracks[-1]))
    

#--  PROCESSING  --------------------------------------------------------------#

# max projection
img_dapi_zmax = np.amax(img_dapi, axis=0)
img_gfp_zmax = np.amax(img_gfp, axis=0)
img_fish_zmax = np.amax(img_fish, axis=0)

# output image in plot
fig, ax = plt.subplots(1, 3, figsize=(10,4))
ax[0].imshow(img_dapi_zmax, cmap=su.cmap_KtoB)
ax[1].imshow(img_gfp_zmax, cmap=su.cmap_KtoG)
ax[2].imshow(img_fish_zmax, cmap=su.cmap_KtoR)
plt.savefig(os.path.join(outdir, img_name + '_channels.png'), dpi=300)
plt.close()

# segment nuclei
print('\nSegmenting nuclei...', end='')
with silence_stdout():
    masks_nuc, _, _, _ = model_nuc.eval([img_dapi_zmax], rescale=None, channels=[[0, 0]], diameter=80.)
    labeled_mask_nuc = masks_nuc[0]
    labeled_mask_nuc = morphology.remove_small_objects(labeled_mask_nuc, 256)
nuclei_labels = sorted(list(set(list(labeled_mask_nuc.flatten()))))[1:]
nuclei_coords = {l:[tuple(row) for row in np.argwhere(labeled_mask_nuc == l)] for l in nuclei_labels}
print('Done.')

# identify FISH-positive cells and quantify intensity
bg_intens = np.median(img_fish_zmax[labeled_mask_nuc == 0])
positive_labels = []
intensities = []
positive_mask = np.zeros_like(labeled_mask_nuc)
for l in nuclei_labels:
    nuc_intens = np.mean(img_fish_zmax[labeled_mask_nuc == l])
    if nuc_intens/bg_intens > 1.5:  # positive
        positive_labels.append(l)
        intensities.append(nuc_intens-bg_intens)
        positive_mask = positive_mask + (labeled_mask_nuc == l).astype(int)

negative_mask = (labeled_mask_nuc > 0).astype(int) - positive_mask

# output segmentation
fig, ax = plt.subplots(1, 3, figsize=(10,4))
ax[0].imshow(img_dapi_zmax, cmap=su.cmap_KtoB)
ax[1].imshow(img_fish_zmax, cmap=su.cmap_KtoR)
ax[2].imshow(negative_mask, cmap='binary_r')
ax[2].imshow(positive_mask, cmap=su.cmap_NtoR)
plt.savefig(os.path.join(outdir, img_name + '_mask.png'), dpi=300)
plt.close()

# output intensities
with open(os.path.join(outdir, img_name + '_intensities.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows([intensities])

