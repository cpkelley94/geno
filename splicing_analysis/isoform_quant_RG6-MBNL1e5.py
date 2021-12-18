from matplotlib import pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.integrate
import argparse
import os
import csv
import copy

def find_closest_index(l, val):
    '''Return the index of the list item closest to a given value.
    '''
    return min(range(len(l)), key=lambda i: abs(l[i]-val))

# globals
THRESHOLD = 50.

# define isoform lengths and integration bounds (bp)
RG6_EXC_LEN = 327.  # RG6-MBNL1 exon 5 minigene, exclusion product
RG6_EXC_LEFT_BOUND = 316.
RG6_EXC_RIGHT_BOUND = 356.

RG6_INC_LEN = 382.  # RG6-MBNL1 exon 5 minigene, inclusion product
RG6_INC_LEFT_BOUND = 370.
RG6_INC_RIGHT_BOUND = 410.

bounds_inc = [RG6_INC_LEFT_BOUND, RG6_INC_RIGHT_BOUND]
bounds_exc = [RG6_EXC_LEFT_BOUND, RG6_EXC_RIGHT_BOUND]
len_inc = RG6_INC_LEN
len_exc  = RG6_EXC_LEN

# define arguments and parse from command line
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folderpath', help='Path to FA raw data folder.')
parser.add_argument('-o', '--outfile', help='Name of output file.', default='out')
parser.add_argument('-p', '--plotdir', help='Path to the directory for output plots.', default=os.getcwd())
args = vars(parser.parse_args())
p_folder = args['folderpath']
outname = args['outfile']
plotdir = args['plotdir']

# initialize output table and iteration counter
splicing_data = []
n_sample = 0

# iterate through the rows
folder_contents = os.listdir(p_folder)
rows = sorted([item for item in folder_contents if item.startswith('row')])
for rowname in rows:

    # open the file and get the data and headers
    csv_name = [f for f in os.listdir(os.path.join(p_folder, rowname)) if 'Electropherogram' in f][0]
    p_csv = os.path.join(p_folder, rowname, csv_name)

    with open(p_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        indata = np.array([row for row in reader])

    headers = indata[0][1:-1]  # skip x values and ladder
    xdata = indata[1:,0].astype('float64')
    ydata = indata[1:,1:-1].astype('float64')

    # iterate through the samples
    for k, header in enumerate(headers):

        # get sample name from header
        print k, header.split(' ')[1]
        samplename = '_'.join(header.split(' ')[1:])

        # identify and subtract flat baseline
        # (later: should replace with low pass filter method)
        baseline_idx = int(len(ydata[:,k])/2.)
        baseline = sorted(copy.deepcopy(ydata[:,k]))[baseline_idx]
        y_corr = copy.deepcopy(ydata[:,k]) - baseline 

        # find max fluorescence intensity value for each isoform
        xmin_exc = find_closest_index(xdata, bounds_exc[0])
        xmax_exc = find_closest_index(xdata, bounds_exc[1])
        xmin_inc = find_closest_index(xdata, bounds_inc[0])
        xmax_inc = find_closest_index(xdata, bounds_inc[1])

        max_exc = max(list(y_corr)[xmin_exc:xmax_exc])
        max_inc = max(list(y_corr)[xmin_inc:xmax_inc])

        if max_exc < THRESHOLD or max_inc < THRESHOLD:
            # filter out low quality data
            splicing_data.append([rowname[-1], k+1, samplename] + [np.nan]*5)
            plot_integration = False
        else:
            # interpolate the data and integrate the peaks
            electrofunc = scipy.interpolate.interp1d(xdata, y_corr)
            exc_bymass, exc_err = scipy.integrate.quad(electrofunc, bounds_exc[0], bounds_exc[1])[:2]
            inc_bymass, inc_err = scipy.integrate.quad(electrofunc, bounds_inc[0], bounds_inc[1])[:2]

            exc_bymol = exc_bymass/len_exc
            inc_bymol = inc_bymass/len_inc
            psi = inc_bymol/(inc_bymol + exc_bymol)

            splicing_data.append([rowname[-1], k+1, samplename, exc_bymass, inc_bymass, exc_bymol, inc_bymol, psi])
            plot_integration = True

        # draw electropherogram with shaded integration region
        fig, ax = plt.subplots()
        ax.plot(xdata, y_corr, 'k-', linewidth=0.8)
        if plot_integration:
            xfill_exc = np.linspace(bounds_exc[0], bounds_exc[1], num=100)
            yfill_exc = [electrofunc(x) for x in xfill_exc]
            xfill_inc = np.linspace(bounds_inc[0], bounds_inc[1], num=100)
            yfill_inc = [electrofunc(x) for x in xfill_inc]
            ax.fill_between(xfill_exc, yfill_exc, color="#ff5555")
            ax.fill_between(xfill_inc, yfill_inc, color="#5555ff")
        ax.set_xscale('log')
        ax.set_xlim([10, 10000])
        ax.set_xlabel('Size (bp)')
        ax.set_ylabel('Fluorescence intensity')
        plt.savefig(os.path.join(plotdir, samplename + '.pdf'), dpi=300)
        plt.close()

        n_sample += 1  # increment sample counter

# output data to CSV
with open(os.path.join(p_folder, outname + '_rg6-mbnl1e5.csv'), 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows([['row', 'lane', 'sample_name', 'exc (by mass)', 'inc (by mass)', 'exc (by mol)', 'inc (by mol)', 'psi']] + splicing_data)
