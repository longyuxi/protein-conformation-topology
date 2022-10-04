# Calculates the pairwise opposition homology and stores it as a 4D np array
# Usage: python calculate_pairwise_opposition_homologies_binned.py --protein protein_file.pdb --ligand ligand_file.mol2 --output pl_opposition.pckl

import numpy as np
from dataset import get_mol2_coordinates_by_element, get_pdb_coordinates_by_element
from gtda.homology import VietorisRipsPersistence
import argparse
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(description='Calculate persistence homology for up to dimension 2 using opposition distance')
parser.add_argument('--protein', type=str, help='protein file (pdb)')
parser.add_argument('--ligand', type=str, help='ligand file (mol2)')
parser.add_argument('--output', type=str, help='Pickle file to output to')

args = parser.parse_args()
protein_pdb = Path(args.protein)
ligand_mol2 = Path(args.ligand)
output_file = Path(args.output)
# stored into a numpy array

def atom_persistence_homology(coords):
    # Track connected components, loops, and voids
    homology_dimensions = [0, 1, 2]

    # Collapse edges to speed up H2 persistence calculation!
    persistence = VietorisRipsPersistence(
        homology_dimensions=homology_dimensions,
        collapse_edges=True,
        max_edge_length=200
    )

    diagrams_basic = persistence.fit_transform(coords[None, :, :])

    return diagrams_basic

def bin_persistence_diagram(diagram, bins=np.arange(0, 50, 0.25), types_of_homologies=3):
    output = np.zeros((len(bins) + 1, types_of_homologies, 3), dtype=np.int32)

    if diagram is None:
        return output

    # Bin persistence diagram
    # Binned persistence diagram be in the shape of (len(bins) + 1, types_of_homologies, 3)
    #   where the first channel corresponds to each entry,
    #   second channel is corresponds to type of homology (Betti-0, Betti-1, ...)
    #   third channel corresponds to (birth, persist, death)

    diagram = diagram[0]
    # Now diagram should be in shape (n, 3) where each row is
    #   (birth, death, type_of_homology)

    assert len(np.unique(diagram[:, 2])) == types_of_homologies

    for entry in diagram:
        homology_type = int(entry[2])
        # mark the beginning and end bins
        begin_bin = np.digitize(entry[0], bins)
        output[begin_bin, homology_type, 0] += 1
        end_bin = np.digitize(entry[1], bins)
        output[end_bin, homology_type, 2] += 1

        # mark the middle bins if there are any
        if not begin_bin == end_bin:
            output[np.arange(begin_bin + 1, end_bin), homology_type, 1] += 1

    return output

def run(pdb_file, mol2_file, output_file):
    binned_diagrams = []
    # Make pairwise opposition homologies

    def concatenate_coordinates(list_of_coordinates):
        # input: list of ndarray of size (*, 3)
        output = None
        for i in range(len(list_of_coordinates) - 1):
            if i == 0:
                output = np.concatenate((list_of_coordinates[i], list_of_coordinates[i+1]))
            else:
                output = np.concatenate((output, list_of_coordinates[i+1]))

        return output

    protein_heavy_elements = ['C', 'N', 'O', 'S']
    ligand_heavy_elements = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']

    homologies = [] # this is used to store all of the calculated persistence diagrams

    # 2: all heavy atoms of protein
    protein_heavy_atom_coords = []
    for pe in protein_heavy_elements:
        protein_coords = get_pdb_coordinates_by_element(pdb_file, pe)
        protein_heavy_atom_coords.append(protein_coords)

    protein_heavy_atom_coords = concatenate_coordinates(protein_heavy_atom_coords)

    homologies.append(bin_persistence_diagram(atom_persistence_homology(protein_heavy_atom_coords)))

    # 3: all heavy atoms of protein and all heavy atoms of ligand
    ligand_heavy_atom_coords = []
    for le in ligand_heavy_elements:
        ligand_coords = get_mol2_coordinates_by_element(mol2_file, le)
        ligand_heavy_atom_coords.append(ligand_coords)

    ligand_heavy_atom_coords = concatenate_coordinates(ligand_heavy_atom_coords)
    all_heavy_atom_coords = np.concatenate((protein_heavy_atom_coords, ligand_heavy_atom_coords))

    homologies.append(bin_persistence_diagram(atom_persistence_homology(all_heavy_atom_coords)))

    # 4: all carbon atoms of protein
    protein_carbon_coords = get_pdb_coordinates_by_element(pdb_file, 'C')
    homologies.append(bin_persistence_diagram(atom_persistence_homology(protein_carbon_coords)))

    # 5: all carbon atoms of protein and all heavy atoms of ligand
    ligand_carbon_coords = get_mol2_coordinates_by_element(mol2_file, 'C')
    all_carbon_coords = np.concatenate((protein_carbon_coords, ligand_carbon_coords))
    homologies.append(bin_persistence_diagram(atom_persistence_homology(all_carbon_coords)))

    with open(output_file, 'wb') as of:
        pickle.dump(homologies, of)


if __name__ == '__main__':
    run(protein_pdb, ligand_mol2, output_file)