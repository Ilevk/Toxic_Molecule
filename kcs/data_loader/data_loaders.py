import pandas as pd
from os.path import join
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
from kcs.utils import *


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
            data_dir = join(data_dir, 'train_.csv')
        else:
            data_dir = join(data_dir, 'valid_.csv')

        df = pd.read_csv(join(data_dir))
        self.smiles = df["SMILES"]
        self.exp = df["label"].values

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
        return (self.X[index], self.A[index]), self.exp[index]


