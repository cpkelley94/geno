import argparse
import os
import subprocess

# create parser and parse arguments passed in the command line
parser = argparse.ArgumentParser()
parser.add_argument('file_table', type=str, nargs=1, help='Path to the .txt file listing all CZI images and arguments.')
parser.add_argument('outdir', type=str, nargs=1, help='Path to the output directory (necessary subdirectories will not be created).')
parser.add_argument('--run', action='store_true', help='Start SLURM processes.')

args = vars(parser.parse_args())
p_ftable = args['file_table'][0]
outdir = args['outdir'][0]
should_run = args['run']

# file list should be in format [img_path, out_dir, gene1, gene2, t_dapi, t_fiber, t_spot1, t_spot2]
with open(p_ftable, 'r') as listfile:
    file_list = listfile.readlines()

# create a bash script to analyze each image
sh_paths = []
for i, line in enumerate(file_list):
    p_img, t_dapi = line.replace('\r','').replace('\n','').split('\t')
    img_name = os.path.splitext(os.path.basename(p_img))[0]

    cmd = 'module load python/3.6.5\n'
    cmd += 'python analyze_e064.py "' + p_img + '" ' + outdir
    if not t_dapi == '.':
        cmd += ' -d ' + t_dapi
    # if not t_spot_hcr == '.':
    #     cmd += ' -r ' + t_spot_hcr
    cmd += ' --plot\n'

    sh_path = outdir + '_' + str(i).zfill(2) + '.sh'
    sh_paths.append(['e064_' + str(i).zfill(2), sh_path])

    with open(sh_path, 'w') as sh_file:
        sh_file.write(cmd)

for name, bp in sh_paths:
    cmd = 'chmod +x ' + bp
    subprocess.call(cmd, shell=True)

if should_run:
    # slurmify the bash scripts and queue them in slurm
    for name, bp in sh_paths:
        cmd = 'slurmify ' + bp + ' -n ' + name + ' -m 4 -t 1 --run'
        subprocess.Popen(cmd, shell=True)
