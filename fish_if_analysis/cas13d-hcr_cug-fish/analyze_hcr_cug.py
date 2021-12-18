import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
# from itertools import count
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
from mpl_toolkits import mplot3d
from numpy.linalg import norm
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from skimage import feature, exposure, filters, morphology, measure
from xml.etree import ElementTree
import argparse
import csv
import matplotlib.cm as cm
import numpy as np
import os
import scipy.ndimage as ndi
import trimesh

# custom libraries
import scope_utils3 as su


#--  FUNCTION DECLARATIONS  ---------------------------------------------------#

def open_image_2d(img_path, meta_path=None):
    # determine image filetype and open
    if img_path.lower().endswith('.czi'):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # open CZI format using `czifile` library
        from czifile import CziFile
        with CziFile(img_path) as czi_file:
            img_czi = czi_file.asarray()
            if meta_path is not None:
                metatree = ElementTree.parse(meta_path)
            else:
                meta = czi_file.metadata()
                metatree = ElementTree.fromstring(meta)
        
        img = img_czi[0,0,:,0,0,:,:,0]  # c, x, y

    else:
        raise ValueError('Image filetype not recognized. Allowed: .CZI')

    return img, img_name, metatree

def threshold_nuclei_2d(img_dapi, t_dapi=None, labeled=True, multiplier=0.75, verbose=False):
    if verbose:
        print('\nSegmenting nuclei...')

    # image preprocessing
    sig = (10, 10)

    img_dapi_norm = su.normalize_image(img_dapi)
    img_blur = filters.gaussian(img_dapi_norm, sig)

    # get threshold using method determined from value of t_dapi
    if t_dapi is None:
        thresh = filters.threshold_otsu(img_blur)*multiplier
    elif type(t_dapi) == float:
        thresh = t_dapi
    else:
        raise TypeError('`t_dapi` argument not recognized. \
            Must be either float or None.')
    
    if verbose:
        print('DAPI threshold = ' + str(thresh))
    
    # binarize and clean up mask
    bin_dapi = np.where(img_blur > thresh, 1, 0)
    bin_dapi = morphology.remove_small_objects(bin_dapi.astype(bool), 2048)
    bin_dapi = morphology.remove_small_holes(bin_dapi.astype(bool), 2048)
    nuclei_labeled, n_nuc = morphology.label(bin_dapi, return_num=True)

    if labeled:
        return nuclei_labeled
    else:
        return np.where(nuclei_labeled > 0, 1, 0)

def find_spots_2d(img_fish, sigma=1., t_spot=5, mask=None):
    spots = feature.blob_log(img_fish, sigma, sigma, num_sigma=1, threshold=t_spot)
    
    if mask is not None:
        spots_masked = []
        for spot in spots:
            spot_pos = tuple(spot[0:2].astype(int))
            if mask[spot_pos]:
                spots_masked.append(spot)
        spots_masked = np.row_stack(spots_masked)
        return spots_masked
    else:
        return spots


#--  COMMAND LINE ARGUMENTS  --------------------------------------------------#

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('img', type=str, nargs=1, help='Path to image file (CZI).')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('-d', '--dapi-threshold', help='Threshold value for nucleus segmentation (default: Otsu\'s method)', default=None)
parser.add_argument('-r', '--spot-threshold-hcr', help='Threshold value for spot detection in Cas13d HCR-FISH channel (default: 0.006)', default=0.006)
parser.add_argument('-c', '--spot-threshold-cug', help='Threshold value for spot detection in CUG FISH channel (default: 0.002)', default=0.002)
parser.add_argument('--plot', action='store_true', help='Generate plots, images, and animations.')

# parse arguments
args = vars(parser.parse_args())
p_img = args['img'][0]
outdir = args['outdir'][0]
should_plot = args['plot']
t_dapi = args['dapi_threshold']
t_spot_hcr = args['spot_threshold_hcr']
t_spot_cug = args['spot_threshold_cug']

if t_dapi:
    t_dapi = float(t_dapi)
if t_spot_hcr is not None:
    t_spot_hcr = float(t_spot_hcr)
if t_spot_cug is not None:
    t_spot_cug = float(t_spot_cug)
t_spots = [t_spot_hcr, t_spot_cug]


#--  INPUT FILE OPERATIONS  ---------------------------------------------------#

# determine image filetype and open
print('Analyzing `' + p_img + '`...\n')

img, img_name, mtree = open_image_2d(p_img)


img_hcr, img_cug, img_cellmask, img_dapi = img
vmaxs = [15, 5, 10, 5] 

if should_plot:
    su.draw_images([img_hcr, img_cug, img_cellmask, img_dapi], titles=['Cas13d HCR FISH', 'CUG FISH', 'CellMask', 'DAPI'], vmax=vmaxs, out_name=os.path.join(outdir, 'images', img_name+'_channels.pdf'))

# get pixel dimensions
dim_iter = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
dims = {}
for dimension in dim_iter:
    dim_id = dimension.get('Id')
    dims.update({dim_id:float(dimension.find('Value').text)*1.E6}) # [um]
pixel_area = dims['X'] * dims['Y']  # [um^3]
dims_xy = np.array([dims['X'], dims['Y']])

# get channel wavelengths
track_tree = mtree.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup').findall('TrackSetup')
tracks = []
for track in track_tree:
    name = track.get('Name')
    wavelen = int(float(track.find('Attenuators').find('Attenuator').find('Wavelength').text)*1E9)
    tracks.append(wavelen)

print('Tracks: ' + ', '.join(map(str, tracks)))


#--  SEGMENTATION  ------------------------------------------------------------#

nuclei_labeled = threshold_nuclei_2d(img_dapi, t_dapi=t_dapi, verbose=True)
nuclei_binary = np.where(nuclei_labeled > 0, 1, 0)
nuclei_border = morphology.binary_dilation(nuclei_binary, selem=morphology.disk(2)) - nuclei_binary

if should_plot:
    su.draw_images([nuclei_binary, img_dapi], vmax=[1, 5], titles=['nuclei', 'DAPI'], cmaps=['binary_r', su.cmap_KtoB], out_name=os.path.join(outdir, 'images', img_name+'_regions.pdf'))


#--  FISH SPOT DETECTION  -----------------------------------------------------#

hcr_spots_masked = find_spots_2d(img_hcr, t_spot=t_spot_hcr, sigma=2.)
print(str(hcr_spots_masked.shape[0]) + ' HCR FISH spots detected.')

if should_plot:
    fig, ax = plt.subplots()
    im_xy = ax.imshow(img_hcr, vmax=15, cmap='binary_r')
    im_nucborder = ax.imshow(nuclei_border, vmax=1, cmap=su.cmap_NtoB)
    spots_xy = ax.scatter(hcr_spots_masked[:,1], hcr_spots_masked[:,0], s=12, c='none', edgecolors='r', linewidths=0.5, alpha=0.5)
    plt.savefig(os.path.join(outdir, 'images', img_name + '_hcr_detection.pdf'), dpi=600)
    plt.close()

cug_spots_masked = find_spots_2d(img_cug, t_spot=t_spot_cug, sigma=2.)
print(str(cug_spots_masked.shape[0]) + ' CUG FISH spots detected.')

if should_plot:
    fig, ax = plt.subplots()
    im_xy = ax.imshow(img_cug, vmax=5, cmap='binary_r')
    im_nucborder = ax.imshow(nuclei_border, vmax=1, cmap=su.cmap_NtoB)
    spots_xy = ax.scatter(cug_spots_masked[:,1], cug_spots_masked[:,0], s=12, c='none', edgecolors='y', linewidths=0.5, alpha=0.5)
    plt.savefig(os.path.join(outdir, 'images', img_name + '_cug_detection.pdf'), dpi=600)
    plt.close()


#--  PER-NUCLEUS FISH CALCULATIONS  -------------------------------------------#

hcr_spot_table = [['#nucleus', 'x', 'y', 'intensity']]
hcr_nuc_table = [['#nucleus', 'mean', 'total']]
cug_spot_table = [['#nucleus', 'x', 'y', 'intensity']]
cug_nuc_table = [['#nucleus', 'mean', 'total']]

for l in sorted(np.unique(nuclei_labeled)):  # 0 is outside nuclei, 1+ is within nuclei
    this_nucleus = (nuclei_labeled == l)

    # find HCR FISH spots in this nucleus
    these_hcr_spots = []
    for hcr_spot in hcr_spots_masked:
        if this_nucleus[tuple(hcr_spot[:2].astype(int))]:
            these_hcr_spots.append(hcr_spot)  # spot is in the nucleus, count it

    # calculate spot intensities
    img_hcr_blur = filters.gaussian(img_hcr, (2,2), preserve_range=True)
    hcr_spot_intensities = [img_hcr_blur[tuple(spot[:2].astype(int))] for spot in these_hcr_spots]

    # calculate total HCR FISH intensity
    pix_intens = img_hcr[this_nucleus]
    nuclear_mean_fish = np.mean(pix_intens)
    nuclear_total_fish = np.sum(pix_intens)

    # append to output tables
    if l == 0:  # cytoplasm
        for spot, intens in zip(these_hcr_spots, hcr_spot_intensities):
            hcr_spot_table.append([img_name + '-cyt'] + list(spot[:2]) + [intens])
    else:  # nucleus
        for spot, intens in zip(these_hcr_spots, hcr_spot_intensities):
            hcr_spot_table.append([img_name + '-' + str(l)] + list(spot[:2]) + [intens])
        hcr_nuc_table.append([img_name + '-' + str(l), nuclear_mean_fish, nuclear_total_fish])
    
    # find CUG FISH spots in this nucleus
    these_cug_spots = []
    for cug_spot in cug_spots_masked:
        if this_nucleus[tuple(cug_spot[:2].astype(int))]:
            these_cug_spots.append(cug_spot)  # spot is in the nucleus, count it
    # these_cug_spots = np.stack(these_cug_spots, axis=0)

    # calculate spot intensities
    img_cug_blur = filters.gaussian(img_cug, (2,2), preserve_range=True)
    cug_spot_intensities = [img_cug_blur[tuple(spot[:2].astype(int))] for spot in these_cug_spots]

    # calculate total CUG FISH intensity
    bg = 2  # set threshold for background signal intensity
    img_cug_baseline = img_cug.astype(np.int32) - bg
    img_cug_baseline[img_cug_baseline < 0] = 0
    pix_intens = img_cug_baseline[this_nucleus]
    nuclear_mean_fish = np.mean(pix_intens)
    nuclear_total_fish = np.sum(pix_intens)

    # append to output tables
    if l == 0:  # cytoplasm
        for spot, intens in zip(these_cug_spots, cug_spot_intensities):
            cug_spot_table.append([img_name + '-cyt'] + list(spot[:2]) + [intens])
    else:  # nucleus
        for spot, intens in zip(these_cug_spots, cug_spot_intensities):
            cug_spot_table.append([img_name + '-' + str(l)] + list(spot[:2]) + [intens])
        cug_nuc_table.append([img_name + '-' + str(l), nuclear_mean_fish, nuclear_total_fish])
    
    # if should_plot:
    #     fig, ax = plt.subplots()
    #     im_xy = ax.imshow(img_cug_baseline, vmax=4, cmap='binary_r')
    #     im_nucborder = ax.imshow(nuclei_border, vmax=1, cmap=su.cmap_NtoB)
    #     spots_xy = ax.scatter(these_cug_spots[:,1], these_cug_spots[:,0], s=12, c='none', edgecolors='y', linewidths=0.5, alpha=0.5)
    #     plt.savefig(os.path.join(outdir, 'images', img_name + '_cug_nucleus.pdf'), dpi=600)
    #     plt.close()

# output tables to file
with open(os.path.join(outdir, 'hcr_spot_intensity', img_name + '_hcr_intensities.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(hcr_spot_table)

with open(os.path.join(outdir, 'hcr_total_signal', img_name + '_hcr_total_signal.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(hcr_nuc_table)

with open(os.path.join(outdir, 'cug_spot_intensity', img_name + '_cug_intensities.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(cug_spot_table)

with open(os.path.join(outdir, 'cug_total_signal', img_name + '_cug_total_signal.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(cug_nuc_table)
    


    
    
    
