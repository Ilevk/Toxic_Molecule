import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class GCNNet(BaseModel):

    def __init__(self,
                 in_dim,
                 out_dim,
                 molvec_dim,
                 n_layer,
                 use_bn,
                 gcn_activation,
                 mlp_activation,
                 drop_rate,
                 vocab_size,
                 degree_size,
                 numH_size,
                 valence_size,
                 isarom_size,
                 emb_train,
                 mlp_dims):
        super(GCNNet, self).__init__()

        # Create Atom Element embedding layer
        self.embedding = self.create_emb_layer([vocab_size, degree_size,
                                                numH_size, valence_size,
                                                isarom_size], emb_train)
        self.gcn_act = self.select_activation(gcn_activation)
        self.mlp_act = self.select_activation(mlp_activation)
        
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layers.append(GConv(in_dim if i == 0 else out_dim,
                                         out_dim,
                                         use_bn,
                                         self.gcn_act,
                                         drop_rate))

        self.readout = Readout(out_dim, molvec_dim//4)

        self.fc1 = nn.Linear(molvec_dim, molvec_dim // 2)
        self.fc2 = nn.Linear(molvec_dim // 2, molvec_dim // 2)
        self.fc3 = nn.Linear(molvec_dim // 2, 2)
        self.bn1 = BN1d(molvec_dim, use_bn)
        self.bn2 = BN1d(molvec_dim // 2, use_bn)
        self.bn3 = BN1d(molvec_dim // 2, use_bn)
        self.dp  = nn.Dropout(drop_rate)
        
        self.MLP1 = MLP(mlp_dims, self.mlp_act, drop_rate)
        self.MLP2 = MLP(mlp_dims, self.mlp_act, drop_rate)
        self.MLP3 = MLP(mlp_dims, self.mlp_act, drop_rate)
        
    def create_emb_layer(self, list_vocab_size, emb_train=False):
        list_emb_layer = nn.ModuleList()
        for i, vocab_size in enumerate(list_vocab_size):
            vocab_size += 1
            emb_layer = nn.Embedding(vocab_size, vocab_size)
            weight_matrix = torch.zeros((vocab_size, vocab_size))
            for i in range(vocab_size):
                weight_matrix[i][i] = 1
            emb_layer.load_state_dict({'weight': weight_matrix})
            emb_layer.weight.requires_grad = emb_train
            list_emb_layer.append(emb_layer)
        return list_emb_layer

    def _embed(self, x):
        list_embed = list()
        for i in range(5):
            list_embed.append(self.embedding[i](x[:, :, i]))
        x = torch.cat(list_embed, 2)
        return x

    def forward(self, x, A, m1_x, m2_x, m3_x):
        
        # GCN
        A = A.float()
        x = self._embed(x)

        for i, module in enumerate(self.gcn_layers):
            x, A = module(x, A)
        gcn_x = self.readout(x)
        
        # MLP
        mlp1_x = self.MLP1(m1_x)
        mlp2_x = self.MLP2(m2_x)
        mlp3_x = self.MLP3(m3_x)
        
        x = torch.cat((gcn_x, mlp1_x, mlp2_x, mlp3_x), dim=1)
        
#         x = self.bn1(x)
        x = self.mlp_act(self.fc1(x))
        x = self.dp(x)
        x = self.bn2(x)
        x = self.mlp_act(self.fc2(x))
        x = self.dp(x)
        x = self.bn3(x)
        x = self.fc3(x)
        
        return torch.squeeze(x)

    def select_activation(self, act):
        if act   == 'relu':
            return nn.ReLU()
        elif act == 'lrelu':
            return nn.LeakyReLU()
        elif act == 'gelu':
            return nn.GELU()


class BN1d(nn.Module):
    def __init__(self, out_dim, use_bn):
        super(BN1d, self).__init__()
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        if not self.use_bn:
            return x
        origin_shape = x.shape
        x = x.view(-1, origin_shape[-1])
        x = self.bn(x)
        x = x.view(origin_shape)
        return x


class GConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn, act, drop_rate = 0.5):
        super(GConv, self).__init__()
        self.fc  = nn.Linear(input_dim, output_dim)
        self.bn  = BN1d(output_dim, use_bn)
        self.dp  = nn.Dropout(drop_rate)
        self.act = act

    def forward(self, X, A):
        x = self.fc(X)
        x = torch.matmul(A, x)
        x = self.bn(x)
        x = self.dp(x)
        x = self.act(x)
        return x, A


class Readout(nn.Module):
    def __init__(self, out_dim, molvec_dim):
        super(Readout, self).__init__()
        self.readout_fc = nn.Linear(out_dim, molvec_dim)
        nn.init.xavier_normal_(self.readout_fc.weight.data)

    def forward(self, output_H):
        molvec = self.readout_fc(output_H)
        molvec = torch.mean(molvec, dim=1)
        return molvec
    

class MLP(nn.Module):
    def __init__(self, mlp_dims, mlp_act, drop_rate):
        super(MLP, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.BatchNorm1d(mlp_dims[1]),
            mlp_act,
            nn.Dropout(p=drop_rate)
            )

        self.layer2 = nn.Sequential(
            nn.Linear(mlp_dims[1], mlp_dims[2]),
            nn.BatchNorm1d(mlp_dims[2]),
            mlp_act,
            nn.Dropout(p=drop_rate)
            )

        self.layer3 = nn.Sequential(
            nn.Linear(mlp_dims[2], mlp_dims[3]),
            nn.BatchNorm1d(mlp_dims[3]),
            mlp_act,
            nn.Dropout(p=drop_rate)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x