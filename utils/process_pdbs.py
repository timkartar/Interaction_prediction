from Bio.PDB import PDBParser, PDBIO
from tqdm import tqdm
import sys

pdbs=[]
chains_to_keep=[]
for line in [l.strip() for l in open("/project/rohs_102/raktimmi/interactions/data.csv","r").readlines()[1:]]:
    split = line.split(",")[0]
    pdbs.append(split.split("_")[0])
    chains_to_keep.append(split.split("_")[1].replace(":",""))

print(len(pdbs))
print(len(chains_to_keep))

pdbio = PDBIO()
parser = PDBParser()



pdb_path = "/project/rohs_102/raktimmi/interactions/data/raw_pdbs/"
processed_path = "/project/rohs_102/raktimmi/interactions/data/processed/"

for i in range(len(pdbs)):
    structure = parser.get_structure(pdbs[i], pdb_path + pdbs[i].lower() + ".pdb")
    model = structure[0]
    child_list = model.child_list.copy();
    for chain in child_list:
        if chain.id not in chains_to_keep[i]:
            print("removed", pdbs[i], chain.id)
            model.detach_child(chain.id)
    pdbio.set_structure(structure)
    pdbio.save(processed_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".pdb")


