# %%
from dataset import load_pdbbind_data_index
import socket

if 'dcc' in socket.gethostname():
    index_location = '/work/yl708/pdbbind/index/INDEX_refined_data.2020'
elif '1080' in socket.gethostname():
    index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'
else:
    raise Exception # not implemented

index = load_pdbbind_data_index(index_location)

## For debug purposes

index = index[:10]

# %% [markdown]
# ### Calculating the persistence diagram of each of the coordinates and storing them

# %%
from pathlib import Path

# for each item, calculate the persistence homology of it
output_folder = 'computed_homologies'

# get pdb file for both things
base_folder = Path(index_location).parent.parent / 'refined-set'

if 'dcc' in socket.gethostname():
    python_interpreter = '/work/yl708/bass/cycada/.conda/donaldlab/bin/python'
    script_base_path = Path('/work/yl708/protein-conformation-topology')
elif '1080' in socket.gethostname():
    python_interpreter = '/home/longyuxi/miniconda3/envs/donaldlab/bin/python'
    script_base_path = Path('/home/longyuxi/Documents/protein-conformation-topology')
else:
    raise Exception # not implemented

homology_calculator_location = script_base_path / 'calculate_homology.py'
pairwise_opposition_homology_calculator_location = script_base_path / 'calculate_pairwise_opposition_homologies_binned.py'
atom_2345_homology_calculator_location = script_base_path / 'calculate_2345_homology.py'

# list of commands to be run by GNU parallel
# for any of the files, if they don't exist, calculate homology for it

commands = ''
for idx, row in index.iterrows():
    pdb_name = row['PDB code']
    diagram_output_folder = Path(output_folder) / pdb_name
    diagram_output_folder.mkdir(parents=True, exist_ok=True)

    # input files
    pocket_pdb = base_folder / pdb_name / f'{pdb_name}_pocket.pdb'
    protein_pdb = base_folder / pdb_name / f'{pdb_name}_protein.pdb'
    ligand_mol2 = base_folder / pdb_name / f'{pdb_name}_ligand.mol2'

    # # protein
    # if not (diagram_output_folder / 'protein.pckl').is_file():
    #     commands += f'{python_interpreter} {homology_calculator_location} --input {str(protein_pdb)} --output {str(diagram_output_folder / "protein.pckl")}\n'

    # # pocket
    # if not (diagram_output_folder / 'pocket.pckl').is_file():
    #     commands += f'{python_interpreter} {homology_calculator_location} --input {str(pocket_pdb)} --output {str(diagram_output_folder / "pocket.pckl")}\n'

    # # ligand
    # if not (diagram_output_folder / 'ligand.pckl').is_file():
    #     commands += f'{python_interpreter} {homology_calculator_location} --input {str(ligand_mol2)} --output {str(diagram_output_folder / "ligand.pckl")}\n'

    # pairwise opposition homology
    # if not (diagram_output_folder / 'pairwise_opposition.pckl').is_file():
    #     commands += f'{python_interpreter} {pairwise_opposition_homology_calculator_location} --protein {str(protein_pdb)} --ligand {str(ligand_mol2)} --output {str(diagram_output_folder / "pairwise_opposition.pckl")}\n'

    # 2345 homology
    if not (diagram_output_folder / '2345_homology.pckl').is_file():
        commands += f'{python_interpreter} {atom_2345_homology_calculator_location} --protein {str(protein_pdb)} --ligand {str(ligand_mol2)} --output {str(diagram_output_folder / "2345_homology.pckl")}\n'

with open('cmds.tmp', 'w') as cf:
    cf.write(commands)

# Then run the following in terminal (for local)
#   time parallel -j 10 --eta < cmds.tmp

# Or do something fancy (on computing cluster)