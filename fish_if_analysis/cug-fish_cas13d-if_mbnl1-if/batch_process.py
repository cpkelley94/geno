import argparse
import os
import subprocess

# create parser and parse arguments passed in the command line
parser = argparse.ArgumentParser()
parser.add_argument('images', type=str, nargs=1, help='Path to the directory containing all CZI images.')
parser.add_argument('--run', action='store_true', help='Start SLURM processes.')

args = vars(parser.parse_args())
image_dir = args['images'][0]
should_run = args['run']

# get all images
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# create a bash script to analyze each image
sh_paths = []
for i, p_img in enumerate(image_paths):
    img_name = os.path.splitext(os.path.basename(p_img))[0]

    cmd = 'module load python/3.6.5\n'
    cmd += 'python analyze_cug_cas13d_mbnl1.py "' + p_img + '" out'

    sh_path = img_name + '.sh'
    sh_paths.append(['img_' + str(i).zfill(3), sh_path])

    with open(sh_path, 'w') as sh_file:
        sh_file.write(cmd)

for name, bp in sh_paths:
    cmd = 'chmod +x ' + bp
    subprocess.call(cmd, shell=True)

if should_run:
    # slurmify the bash scripts and queue them in slurm
    for name, bp in sh_paths:
        cmd = 'slurmify ' + bp + ' -n ' + name + ' -m 8 -t 1 --run'
        subprocess.Popen(cmd, shell=True)
