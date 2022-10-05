## A script that submits SLURM batch jobs from cmds.tmp, using a template sbatch-template.sh

from pathlib import Path

# Create temporary folder for jobs
temp_sh_folder = Path('sbatch_tmp')
temp_sh_folder.mkdir(parents=True, exist_ok=True)

# Create temporary folder for outputs
temp_output_folder = Path('job_outputs')
temp_output_folder.mkdir(parents=True, exist_ok=True)
temp_output_folder = temp_sh_folder.resolve()

with open('cmds.tmp', 'r') as f:
    lines = f.readlines()

sbatch_template_file = '/work/yl708/protein-conformation-topology/sbatch-template.sh'

with open(sbatch_template_file, 'r') as f:
    sbatch_template = f.read()

# for line in cmds.tmp
    # create a copy of sbatch-template.sh, and in that copy, do
    # echo the line to both stdout and stderr by echo "foo" | tee /dev/stderr
    # replace {COMMAND} with this line

import os

for idx, line in enumerate(lines):
    commands = f'echo "{line}" | tee /dev/stderr\n' + line
    # Replace output placeholders
    out_sbatch_content = sbatch_template.replace('%OUTPUT_FILE%', str(temp_output_folder / f'job_{idx}.out'))
    out_sbatch_content = out_sbatch_content.replace('%ERROR_FILE%', str(temp_output_folder / f'job_{idx}.err'))

    # Replace command placeholders
    out_sbatch_content = out_sbatch_content.replace('%COMMAND%', commands)
    out_sbatch_file = temp_sh_folder / f'job_{idx}.sh'
    with open(out_sbatch_file, 'w') as o:
        o.write(out_sbatch_content)

    # submit job
    os.system(f'sbatch {out_sbatch_file}')

