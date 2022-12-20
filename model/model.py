import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_modal import MULTModel

class Self_Attention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(num_hidden / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        q = self.transpose_for_scores(q)  # [bsz, heads, protein_len, hid]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        if mask is not None:
            attention_mask = (1.0 - mask) * -10000
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(
                1
            )

        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        outputs = torch.matmul(attention_scores, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        return outputs


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.leaky_relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden=64, num_heads=4, dropout=0.2):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList(
            [nn.LayerNorm(num_hidden, eps=1e-6) for _ in range(2)]
        )

        self.attention = Self_Attention(num_hidden, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, mask=None):
        # Self-attention
        dh = self.attention(h_V, h_V, h_V, mask)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask is not None:
            mask = mask.unsqueeze(-1)
            h_V = mask * h_V
        return h_V


class LMetalSite_Test(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        num_encoder_layers=2,
        num_heads=4,
        augment_eps=0.05,
        dropout=0.2,
    ):
        super(LMetalSite, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Embedding layers
        self.input_block = nn.Sequential(
            nn.LayerNorm(feature_dim, eps=1e-6),
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        self.hidden_block = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim, eps=1e-6),
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        # ion-specific layers
        self.FC_ZN1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_ZN2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_CA1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CA2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MG1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MG2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MN1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MN2 = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, protein_feat, mask):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            protein_feat = protein_feat + self.augment_eps * torch.randn_like(
                protein_feat
            )

        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)

        logits_ZN = self.FC_CA2(F.leaky_relu(self.FC_CA1(h_V))).squeeze(-1)
        logits_CA = self.FC_CA2(F.leaky_relu(self.FC_CA1(h_V))).squeeze(-1)
        logits_MG = self.FC_MG2(F.leaky_relu(self.FC_MG1(h_V))).squeeze(-1)
        logits_MN = self.FC_MN2(F.leaky_relu(self.FC_MN1(h_V))).squeeze(-1)
        logits = torch.cat((logits_ZN, logits_CA, logits_MG, logits_MN), 1)

        return logits


class LMetalSite(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        num_encoder_layers=2,
        num_heads=4,
        augment_eps=0.05,
        dropout=0.2,
        ion_type="ZN",
        training=True,
    ):
        super(LMetalSite, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps
        self.training = training
        # Embedding layers
        if (
            feature_dim == 384
        ):  # hard encode a sigmoid for evoformer to replace min-max normalization
            self.input_block = nn.Sequential(
                nn.Sigmoid(),
                nn.LayerNorm(feature_dim, eps=1e-6),
                nn.Linear(feature_dim, hidden_dim),
                nn.LeakyReLU(),
            )
        else:
            self.input_block = nn.Sequential(
                nn.LayerNorm(feature_dim, eps=1e-6),
                nn.Linear(feature_dim, hidden_dim),
                nn.LeakyReLU(),
            )

        self.hidden_block = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim, eps=1e-6),
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        assert ion_type in ["ZN", "CA", "MG", "MN"]
        self.ion_type = ion_type

        # ion-specific layers
        self.ZN_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.CA_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.MG_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.MN_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, protein_feat, mask):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            protein_feat = protein_feat + self.augment_eps * torch.randn_like(
                protein_feat
            )

        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)

        if self.ion_type == "ZN":
            logits = self.ZN_head(h_V).squeeze(-1)
        elif self.ion_type == "CA":
            logits = self.CA_head(h_V).squeeze(-1)
        elif self.ion_type == "MN":
            logits = self.MN_head(h_V).squeeze(-1)
        elif self.ion_type == "MG":
            logits = self.MG_head(h_V).squeeze(-1)

        return logits

class LMetalSiteMultiModal(nn.Module):
    def __init__(
        self,
        conf,
        ion_type="ZN",
        training=True,
    ):
        super(LMetalSiteMultiModal, self).__init__()

        # Hyperparameters
        self.augment_eps = conf.model.augment_eps
        hidden_dim = conf.model.hidden_dim
        self.training = training
        self.encoding_module = MULTModel(conf.model)

        assert ion_type in ["ZN", "CA", "MG", "MN"]
        self.ion_type = ion_type

        # ion-specific layers
        self.ZN_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.CA_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.MG_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.MN_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_a, feat_b):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            feat_a = feat_a + self.augment_eps * torch.randn_like(
                feat_a
            )
            feat_b = feat_b + self.augment_eps * torch.randn_like(
                feat_b
            )
        output, _ = self.encoding_module(feat_a, feat_b)
        # h_V = self.input_block(protein_feat)
        # h_V = self.hidden_block(h_V)

        # for layer in self.encoder_layers:
        #     h_V = layer(h_V, mask)

        if self.ion_type == "ZN":
            logits = self.ZN_head(output).squeeze(-1)
        elif self.ion_type == "CA":
            logits = self.CA_head(output).squeeze(-1)
        elif self.ion_type == "MN":
            logits = self.MN_head(output).squeeze(-1)
        elif self.ion_type == "MG":
            logits = self.MG_head(output).squeeze(-1)

        return logits

