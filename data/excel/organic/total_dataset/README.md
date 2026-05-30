## Description of the structural data

Unzip the `*.tar.gz` dataset with command `tar -xzvf hiqbind.tar.gz` will get two directories corresponding to to the "small molecule" and "polymer" subset of HiQBind.

```bash
-- raw_data_hiq_sm/
-- raw_data_hiq_poly/
```

In each of the directory, you will see a file structure like this: 
```bash
-- 1a69/
   |-- 1a69_FMB_A_240/
       |-- 1a69_FMB_A_240_ligand.pdb
       |-- 1a69_FMB_A_240_protein.pdb
       |-- 1a69_FMB_A_240_protein_hetatm.pdb
       |-- 1a69_FMB_A_240_hetatm.pdb
       |-- 1a69_FMB_A_240_ligand_refined.sdf
       |-- 1a69_FMB_A_240_protein_refined.pdb
   |-- 1a4m_FMB_B_240/
   |-- 1a4m_FMB_C_240/
-- 1a85/
```

Description of the naming conventions:

+ `1a69`: 4-letter PDB ID
+ `FMB`: Name of the ligand. If the ligand is a polymer, it will be format like "ACE-DIP", where "ACE" is the name of the first residue and "DIP" is the name of the last residue.
+ `A`: Ligand chain ID.
+ `240`: Ligand residue number. If the ligand is a polymer, it will be format like "1-3", where "1" is the residue number of the first residue and "3" is the number of the last residue. Note the residue number may contain insertion code, or be a negative integer or zero.

Description of the files:
+ `*_ligand.pdb`: ligand structure extracted from the original PDB (not processed)
+ `*_protein.pdb`: protein structure extracted from the original PDB (not processed). A protein is defined as chains within 10 angstrom of the ligand structure. 
+ `*_protein_hetatm.pdb`: protein structure with additives (solvents, ions) extracted from the original PDB (not processed). Additives are specified with "HETATM" atoms that are within 4 angstroms of the protein chains.
+ `*_hetatm.pdb`: additives' structure extracted from the original PDB (not processed)
+ `*_ligand_refined.sdf`: refined ligand structures (hydrogen added, correct bond order, better tautomer states/protonataion states) with PDBBind-Opt workflow. 
+ `*_protein_refined.pdb`: refined protein structures (hydrogen added, missing atoms/residues added) with PDBBind-Opt workflow. 


## Description of columns in the metadata csv file:
+ `PDBID`: *string*, 4-letter PDB code
+ `Resolution`: *string or float*, resolution of the crystal structure or "NMR" if the structure is resolved by NMR
+ `Year`: *int*, initial deposit year in PDB database
+ `Ligand Name`: *string*, name of the ligand. If the ligand is a polymer, it will be format like "ACE-DIP", where "ACE" is the name of the first residue and "DIP" is the name of the last residue.
+ `Ligand Chain`: *string*, chain of the ligand.
+ `Ligand Residue Number`: *string*, residue number of the ligand. If the ligand is a polymer, it will be format like "1-3", where "1" is the residue number of the first residue and "3" is the number of the last residue. Note the residue number may contain insertion code, or be a negative integer or zero.
+ `Binding Affinity Measurement`: *string*, "kd", "ki" or "ic50". Note in some sources, binding data is labeled to be "Ka" or "Kb", they are converted to Kd using Ka = 1/Kd.
+ `Binding Affinity Sign`: *string*, could be "=", ">=", "<=" or "~".
+ `Binding Affinity Value`: *float*, value of the binding affinity
+ `Binding Affinity Unit`: *string*, coule be "fM", "pM", "nM", "uM", "mM", "M"
+ `Log Binding Affinity`: *float*, binding affinity in log unit
+ `Binding Affinity Source`: *string*, could be "PDBBind", "BindingMOAD", "BindingDB" or "BioLiP"
+ `Binding Affinity Annotation`: *string*, the annotation in the original source.
+ `Protein UniProtID`: *string*, UniProtID of the proteins, seperated by a comma if the ligand bound to more than one chain.
+ `Protein UniProtName`: *string*, Name of the proteins, separated by a comma if the ligand bound to more than one chain.
+ `Ligand SMILES`: *string*, SMILES of the ligand.
+ `Ligand MW`: *float*, Molecular weight of the ligand.
+ `Ligand LogP`: *float*, LogP value of the ligand computed by RDKit.
+ `Ligand TPSA`: *float*, TPSA value of the ligand computed by RDKit.
+ `Ligand NumRotBond`: *int*, Number of rotatable bonds in the ligand.
+ `Ligand NumHeavyAtoms`: *int*, Number of heavy atoms in the ligand.
+ `Ligand NumHDon`: *int*, Number of hydrogen bond donors in the ligand.
+ `Ligand NumHAcc`: *int*, Number of hydrogen bond acceptors in the ligand.
+ `Ligand QED`: *float*, QED value of the ligand computed by RDKit.