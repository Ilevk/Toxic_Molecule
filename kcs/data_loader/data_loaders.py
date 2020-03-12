import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from os.path import join
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
from utils.util import *


class MolculeDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir

        self.total = pd.read_csv(join(data_dir, 'train.csv'))
        self.MAX_LEN =max(self.total['SMILES'].apply(lambda x: get_num_atom(x)))
        self.LIST_SYMBOLS =list(set.union(*self.total['SMILES'].apply(lambda x: get_unique_atom_symbols(x)).values))

        self.dataset = gcnDataset(self.data_dir, self.LIST_SYMBOLS, training, self.MAX_LEN)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class gcnDataset(Dataset):
    def __init__(self, data_dir, LIST_SYMBOLS, training, MAX_LEN=120):
        if training:
            data_dir = join(data_dir, 'train.csv')
        else:
            data_dir = join(data_dir, 'valid_.csv')

        df = pd.read_csv(join(data_dir))
        self.smiles = df["SMILES"]
        self.exp = df["label"].values
        self.mlp1_x = np.concatenate([df.iloc[:, 1:1025].values,    df.iloc[:, -5:-1].values], axis=1)
        self.mlp2_x = np.concatenate([df.iloc[:, 1025:2049].values, df.iloc[:, -5:-1].values], axis=1)
        self.mlp3_x = np.concatenate([df.iloc[:, 2049:-5].values,   df.iloc[:, -5:-1].values], axis=1)
        
        list_X = list()
        list_A = list()

        for i, smiles in enumerate(self.smiles):
            X, A = mol2graph(smiles, LIST_SYMBOLS, MAX_LEN)
            list_X.append(X)
            list_A.append(A)

        self.X = np.array(list_X, dtype=np.long)
        self.A = np.array(list_A, dtype=np.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index], self.A[index]), self.exp[index], self.mlp1_x[index], self.mlp2_x[index], self.mlp3_x[index]


