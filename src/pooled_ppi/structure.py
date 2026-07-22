
import ast, collections, csv, datetime, functools, glob, gzip, hashlib, inspect, io, itertools, json, math, operator, os, os.path, pickle, random, re, requests, shutil, sqlite3, subprocess, string, sys, warnings, zipfile
import Bio, Bio.PDB, Bio.PDB.mmcifio, Bio.PDB.Polypeptide, Bio.SVDSuperimposer, Bio.SeqUtils
import foldcomp
import numpy as np

def to_pdbstr(struct):
    pdbio = Bio.PDB.PDBIO()
    pdbio.set_structure(struct)
    buf = io.StringIO()
    pdbio.save(buf)
    return buf.getvalue()

def to_fcz(struct, name, path):
    with open(path, 'wb') as fh:
        fh.write(foldcomp.compress(name, to_pdbstr(struct)))

def get_structure(path, only_first=True):
    """Attempt to read a structure using Bio.PDB while transparently handling compression/PDB-vs-CIF"""

    if isinstance(path, Bio.PDB.Structure.Structure):
        return path[0] if only_first else path

    # Check file format - pdb1 is used by rcsb bioassemblies...
    if not hasattr(path, 'endswith') or path.endswith('.pdb') or path.endswith('.pdb1') or path.endswith('.pdb.gz') or path.endswith('.pdb1.gz'):
        parser = Bio.PDB.PDBParser(QUIET=True)
    else:
        parser = Bio.PDB.MMCIFParser(QUIET=True)

    # Check for .gz compresssion
    if hasattr(path, 'endswith') and path.endswith('.gz'):
        with gzip.open(path, 'rt') as fh:
            struct = parser.get_structure(path, fh)
    else:
        struct = parser.get_structure(path, path)

    return struct[0] if only_first else struct

def get_resseq(resid):
    return resid.get_id()[1]

def get_atom_resseq(atom):
    return atom.get_parent().get_id()[1]

def is_atom_CA(a): return a.get_id() == 'CA'

def get_chains(path):
    # Read structure; extract peptide sequences, pLDDT, interface residues
    try:
        struct = get_structure(path)
    except PDBConstructionException:
        print('get_chains - skipping', path)
        return

    def get_seq_(chain):
        return ''.join(Bio.PDB.Polypeptide.protein_letters_3to1.get(residue.resname, '') for residue in chain.get_residues())

    def get_CA_(chain):
        #coords_ = (resid['CA'].get_coord() for resid in chain.get_residues())
        #return np.fromiter(coords_, dtype=np.dtype((float, 3)))
        return {get_resseq(resid): resid['CA'].get_coord() for resid in chain.get_residues()}

    def get_pLDDT_(chain):
        #bfactors_ = (resid['CA'].get_bfactor() for resid in chain.get_residues())
        #return np.fromiter(bfactors_, dtype=float)
        return {get_resseq(resid): resid['CA'].get_bfactor() for resid in chain.get_residues()}

    def get_ifresid_(chain, min_distance=5, min_pLDDT=70):
        chain_atm = []
        other_atm = []
        for atm in Bio.PDB.Selection.unfold_entities(struct, 'A'):
            if atm.get_bfactor() >= min_pLDDT:
                if atm.get_parent().get_parent().get_id() == chain.id:
                    chain_atm.append(atm)
                else:
                    other_atm.append(atm)

        if len(other_atm) > 0:
            other_ns = Bio.PDB.NeighborSearch(other_atm)
            chain_ifatoms = list(filter(lambda atom: len(other_ns.search(center=atom.get_coord(), radius=min_distance, level='A')) > 0, chain_atm))
            return sorted(set(map(get_atom_resseq, chain_ifatoms)))
        else:
            return []

    for chain in Bio.PDB.Selection.unfold_entities(entity_list=struct, target_level='C'):
        yield chain.id, get_seq_(chain), get_CA_(chain), get_pLDDT_(chain), get_ifresid_(chain)

def get_ifresid(path, min_distance=8, min_pLDDT=0, include_resname=True):
    """
    Identify interface residues in AF-Multimer structures by filtering on all-atom intra-chain distance (min_distance), and residue-level model confidence (min_pLDDT)
    """
    struct = get_structure(path)
    try:
        chain1, chain2 = struct.get_chains()
    except ValueError:
        return list(), list()

    # Chain 1/2 atoms, filtered by min_pLDDT
    chain1_atoms = list(filter(lambda a: a.get_bfactor() >= min_pLDDT, Bio.PDB.Selection.unfold_entities(struct[chain1.id], 'A')))
    chain2_atoms = list(filter(lambda a: a.get_bfactor() >= min_pLDDT, Bio.PDB.Selection.unfold_entities(struct[chain2.id], 'A')))
    if len(chain1_atoms) == 0 or len(chain2_atoms) == 0:
        return list(), list()
    # Atoms in chain1/2 within radius of the other chain
    chain1_ns = Bio.PDB.NeighborSearch(chain1_atoms)
    chain2_ns = Bio.PDB.NeighborSearch(chain2_atoms)
    chain1_ifatoms = list(filter(lambda atom: len(chain2_ns.search(center=atom.get_coord(), radius=min_distance, level='A')) > 0, chain1_atoms))
    chain2_ifatoms = list(filter(lambda atom: len(chain1_ns.search(center=atom.get_coord(), radius=min_distance, level='A')) > 0, chain2_atoms))

    def get_resseq(atom):
        return atom.get_parent().get_id()[1]

    def get_resname(atom):
        resname3 = str(atom.get_parent().get_resname()).capitalize()
        return Bio.Data.IUPACData.protein_letters_3to1.get(resname3, 'X')

    # Return resseq positions of the corresponding interface residues
    #def get_resseqs(atoms): return sorted(set(((atom.get_parent().get_id()[1], atom.get_parent().get_resname()) for atom in atoms)))
    def get_resseqs(atoms): return sorted(set((atom.get_parent().get_id()[1] for atom in atoms)))
    def get_resseqs_resnames(atoms): return [ f'{resname}{resseq}' for (resseq, resname) in sorted(set(((get_resseq(atom), get_resname(atom)) for atom in atoms))) ]
    if include_resname:
        return get_resseqs_resnames(chain1_ifatoms), get_resseqs_resnames(chain2_ifatoms)
    else:
        return get_resseqs(chain1_ifatoms), get_resseqs(chain2_ifatoms)
