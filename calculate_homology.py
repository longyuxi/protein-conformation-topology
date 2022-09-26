# Usage: python calculate_homology.py --input sth.(pdb/mol2) --output sth.pckl

import argparse
import pickle
from pathlib import Path

from gtda.homology import VietorisRipsPersistence
from dataset import get_pdb_coordinates, get_mol2_coordinates

parser = argparse.ArgumentParser(description='Calculate persistence homology for up to dimension 2 given pdb or mol2 file')
parser.add_argument('--input', type=str, help='pdb or mol2 input file')
parser.add_argument('--output', type=str, help='Pickle file to output to')

args = parser.parse_args()
input_file = Path(args.input)
output_file = Path(args.output)

# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Collapse edges to speed up H2 persistence calculation!
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    collapse_edges=True,
)

if input_file.suffix[1:] == 'pdb':
    coords = get_pdb_coordinates(input_file)
elif input_file.suffix[1:] == 'mol2':
    coords = get_mol2_coordinates(input_file)
else:
    raise Exception('Input file must be pdb or mol2')

diagrams_basic = persistence.fit_transform(coords[None, :, :])

with open(output_file, 'wb') as of:
    pickle.dump(diagrams_basic, of)