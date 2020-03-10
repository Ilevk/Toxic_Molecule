import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from rdkit import Chem


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

def get_unique_atom_symbols(smi):
    mol = Chem.MolFromSmiles(smi)
    symbols = set([atom.GetSymbol() for atom in mol.GetAtoms()])
    return symbols

def get_num_atom(smi):
    mol = Chem.MolFromSmiles(smi)
    num_atom = len([atom.GetSymbol() for atom in mol.GetAtoms()])
    return num_atom

def atom_feature(atom, LIST_SYMBOLS):
    # 1) atom symbol
    # 2) degree
    # 3) 붙어 있는 수소 원자의 수
    # 4) Valence (원자가는 어떤 원자가 다른 원자들과 어느 정도 수준으로 공유 결합을 이루는가를 나타내는 척도)
    # 5) Aromatic 여부 (평평한 고리 구조를 가진 방향족인지)
    return np.array(char_to_ix(atom.GetSymbol(), LIST_SYMBOLS) +
                    char_to_ix(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    char_to_ix(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(int(atom.GetIsAromatic()), [0, 1]))    # (40, 6, 5, 6, 2)

def char_to_ix(x, allowable_set):
    if x not in allowable_set:
        return [0] # Unknown Atom Token
    return [allowable_set.index(x)+1]

def mol2graph(smi, LIST_SYMBOLS, MAX_LEN):
    # MAX_LEN 최대 원자 수
    mol = Chem.MolFromSmiles(smi)

    X = np.zeros((MAX_LEN, 5), dtype=np.uint8)
    A = np.zeros((MAX_LEN, MAX_LEN), dtype=np.uint8)

    temp_A = Chem.rdmolops.GetAdjacencyMatrix(mol
                                             ).astype(np.uint8, copy=False)[:MAX_LEN, :MAX_LEN]
    num_atom = temp_A.shape[0]
    eye_matrix = np.eye(temp_A.shape[0], dtype=np.uint8)
    A[:num_atom, :num_atom] = temp_A + eye_matrix
    
    for i, atom in enumerate(mol.GetAtoms()):
        feature = atom_feature(atom, LIST_SYMBOLS)
        X[i, :] = feature
        if i + 1 >= num_atom: break
            
    return X, A