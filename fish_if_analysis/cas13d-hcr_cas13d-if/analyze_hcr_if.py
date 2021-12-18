from skimage import filters, morphology
from xml.etree import ElementTree
import argparse
import csv
import numpy as np
import os

# custom libraries
import scope_utils3 as su


#--  FUNCTION DECLARATIONS  ---------------------------------------------------#

def open_image_2d(img_path, meta_path=None):
    '''Open 2D CZI image as a numpy array (dimensions: channel, x, y). Return 
    the numpy array, the name of the image, and an ElementTree object containing 
    the microscopy metadata.
    '''

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
    '''Segment nuclei from DAPI channel. Return a binary mask.
    
    If labeled == True, output a mask where each nucleus is labeled with an 
    integer value.
    '''

    if verbose:
        print('\nSegmenting nuclei...')

    # image preprocessing
    sig = (5, 5)

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


#--  COMMAND LINE ARGUMENTS  --------------------------------------------------#

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('img', type=str, nargs=1, help='Path to image file (CZI).')
parser.add_argument('outdir', type=str, nargs=1, help='Directory to export data.')
parser.add_argument('-d', '--dapi-threshold', help='Threshold value for nucleus segmentation (default: Otsu\'s method)', default=None)
parser.add_argument('--plot', action='store_true', help='Generate plots, images, and animations.')

# parse arguments
args = vars(parser.parse_args())
p_img = args['img'][0]
outdir = args['outdir'][0]
should_plot = args['plot']
t_dapi = args['dapi_threshold']

if t_dapi:
    t_dapi = float(t_dapi)


#--  INPUT FILE OPERATIONS  ---------------------------------------------------#

# determine image filetype and open
print('Analyzing `' + p_img + '`...\n')
img, img_name, mtree = open_image_2d(p_img)
img_hcr, img_if, img_cellmask, img_dapi = img  # split channels into separate 2D arrays

# draw all channels and save as PDF
if should_plot:
    su.draw_images([img_hcr, img_if, img_cellmask, img_dapi], titles=['Cas13d HCR FISH', 'HA IF', 'CellMask', 'DAPI'], vmax=[7500, 1750, 15000, 3000], out_name=os.path.join(outdir, 'images', img_name+'_channels.pdf'))

# get pixel dimensions from CZI metadata
dim_iter = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
dims = {}
for dimension in dim_iter:
    dim_id = dimension.get('Id')
    dims.update({dim_id:float(dimension.find('Value').text)*1.E6}) # [um]
pixel_area = dims['X'] * dims['Y']  # [um^2]
dims_xy = np.array([dims['X'], dims['Y']])

# get channel wavelengths from CZI metadata
track_tree = mtree.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup').findall('TrackSetup')
tracks = []
for track in track_tree:
    name = track.get('Name')
    wavelen = int(float(track.find('Attenuators').find('Attenuator').find('Wavelength').text)*1E9)
    tracks.append(wavelen)

print('Tracks: ' + ', '.join(map(str, tracks)))


#--  SEGMENTATION  ------------------------------------------------------------#

# identify nuclei and create binary mask
nuclei_labeled = threshold_nuclei_2d(img_dapi, t_dapi=t_dapi, verbose=True)  # create labeled mask of nuclei
nuclei_binary = np.where(nuclei_labeled > 0, 1, 0)  # create binary mask of nuclei (1 == in a nucleus, 0 == outside nuclei)

# draw segmentation and save as PDF
if should_plot:
    su.draw_images([nuclei_binary, img_dapi], vmax=[1, 3000], titles=['nuclei', 'DAPI'], cmaps=['binary_r', su.cmap_KtoB], out_name=os.path.join(outdir, 'images', img_name+'_regions.pdf'))


#--  PER NUCLEUS CALCULATIONS  ------------------------------------------------#

# initialize output table
nucleus_table = [['#nucleus', 'mean_hcr', 'total_hcr', 'mean_if', 'total_if']]  # headers

# iterate through all nuclei in mask
for l in sorted(np.unique(nuclei_labeled))[1:]:

    # create a mask for this nucleus only
    this_nucleus = (nuclei_labeled == l)

    # calculate total HCR intensity
    pix_intens_hcr = img_hcr[this_nucleus]
    nuclear_mean_hcr = np.mean(pix_intens_hcr)
    nuclear_total_hcr = np.sum(pix_intens_hcr)

    # calculate total IF intensity
    pix_intens_if = img_if[this_nucleus]
    nuclear_mean_if = np.mean(pix_intens_if)
    nuclear_total_if = np.sum(pix_intens_if)

    # add output to table
    nucleus_table.append([img_name + '-' + str(l), nuclear_mean_hcr, nuclear_total_hcr, nuclear_mean_if, nuclear_total_if])

# write output table to file using CSV
with open(os.path.join(outdir, 'nuc_measurements', img_name + '_nuc_measurements.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(nucleus_table)