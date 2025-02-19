import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_modal import MULTModel
from .transformer import TransformerLayer


class LMetalSite_Test(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        num_encoder_layers=2,
        num_heads=4,
        augment_eps=0.05,
        dropout=0.2,
        ligand="ZN",
    ):
        super(LMetalSite_Test, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps
        assert ligand in ["ZN", "CA", "MG", "MN"]
        self.ligand = ligand
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
        # ligand-specific layers
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

        if self.ligand == "ZN":
            logits = self.FC_ZN2(F.leaky_relu(self.FC_ZN1(h_V))).squeeze(-1)
        elif self.ligand == "CA":
            logits = self.FC_CA2(F.leaky_relu(self.FC_CA1(h_V))).squeeze(-1)
        elif self.ligand == "MN":
            logits = self.FC_MN2(F.leaky_relu(self.FC_MN1(h_V))).squeeze(-1)
        elif self.ligand == "MG":
            logits = self.FC_MG2(F.leaky_relu(self.FC_MG1(h_V))).squeeze(-1)

        return logits


class LMetalSiteBase(nn.Module):
    def __init__(self, conf, training=True):
        super(LMetalSiteBase, self).__init__()

        # Hyperparameters
        self.augment_eps = conf.augment_eps
        self.training = training
        self.hidden_dim = conf.hidden_dim
        self.feature_dim = conf.feature_dim
        modules = [
            nn.LayerNorm(self.feature_dim, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
        ]
        self.input_block = nn.Sequential(*modules)
        if conf.fix_encoder:
            self.input_block.requires_grad_(False)
        assert conf.ligand in ["ZN", "CA", "MG", "MN", "DNA", "RNA"]
        self.ligand = conf.ligand

        # ligand-specific layers
        self.ZN_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=True),
        )
        self.CA_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=True),
        )
        self.MG_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=True),
        )
        self.MN_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=True),
        )
        self.DNA_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=True),
        )
        self.RNA_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=True),
        )
        self.params = nn.ModuleDict(
            {
                "encoder": self.input_block,
                "classifier": nn.ModuleList(
                    [
                        self.RNA_head,
                        self.DNA_head,
                        self.MN_head,
                        self.MG_head,
                        self.CA_head,
                        self.ZN_head,
                    ]
                ),
            }
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_noise(self, input):
        if self.training and self.augment_eps > 0:
            input = input + self.augment_eps * torch.randn_like(input)
        return input

    def get_logits(self, input):
        if self.ligand == "ZN":
            logits = self.ZN_head(input).squeeze(-1)
        elif self.ligand == "CA":
            logits = self.CA_head(input).squeeze(-1)
        elif self.ligand == "MN":
            logits = self.MN_head(input).squeeze(-1)
        elif self.ligand == "MG":
            logits = self.MG_head(input).squeeze(-1)
        elif self.ligand == "DNA":
            logits = self.DNA_head(input).squeeze(-1)
        elif self.ligand == "RNA":
            logits = self.RNA_head(input).squeeze(-1)

        return logits

    def forward(self, protein_feat, mask):
        protein_feat = self.add_noise(protein_feat)
        h_V = self.input_block(protein_feat)
        logits = self.get_logits(h_V)

        return logits


class LMetalSiteTwoLayer(LMetalSiteBase):
    def __init__(self, conf, training=True):
        super(LMetalSiteTwoLayer, self).__init__(conf, training=training)
        self.hidden_dim_1 = conf.hidden_dim_1
        self.hidden_dim_2 = conf.hidden_dim_2
        assert self.hidden_dim == self.hidden_dim_2
        self.feature_dim = conf.feature_dim
        modules = [
            nn.LayerNorm(self.feature_dim, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.feature_dim, self.hidden_dim_1),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim_1, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.LeakyReLU(),
        ]
        self.input_block = nn.Sequential(*modules)
        self.params.update({"encoder": self.input_block})


class LMetalSiteLSTM(LMetalSiteBase):
    def __init__(self, conf, training=True):
        super(LMetalSiteLSTM, self).__init__(conf, training=training)
        self.hidden_dim_1 = conf.hidden_dim_1
        self.hidden_dim_2 = conf.hidden_dim_2
        assert self.hidden_dim == self.hidden_dim_2
        self.feature_dim = conf.feature_dim
        modules = [
            nn.LayerNorm(self.feature_dim, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.feature_dim, self.hidden_dim_1),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim_1, eps=1e-6),
            nn.LSTM(
                self.hidden_dim_1,
                hidden_size=self.hidden_dim_2 // 2,
                batch_first=True,
                bidirectional=True,
            ),
        ]
        self.input_block = nn.Sequential(*modules)
        self.params.update({"encoder": self.input_block})

    def forward(self, protein_feat, mask):
        protein_feat = self.add_noise(protein_feat)
        h_V = self.input_block(protein_feat)[0]
        logits = self.get_logits(h_V)

        return logits


class LMetalSite(LMetalSiteBase):
    def __init__(self, conf, training=True):
        super(LMetalSite, self).__init__(conf, training=training)
        hidden_dim = conf.hidden_dim
        self.hidden_block = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim, eps=1e-6),
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerLayer(hidden_dim, conf.num_heads, conf.dropout)
                for _ in range(conf.num_encoder_layers)
            ]
        )

    def forward(self, protein_feat, mask):
        protein_feat = self.add_noise(protein_feat)
        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)

        logits = self.get_logits(h_V)
        return logits


class LMetalSiteMultiModalBase(LMetalSiteBase):
    def __init__(self, conf, training=True):
        super(LMetalSiteMultiModalBase, self).__init__(conf, training=training)
        self.feature_dim_1 = conf.feature_dim_1
        self.feature_dim_2 = conf.feature_dim_2
        self.hidden_dim_1 = conf.hidden_dim_1
        self.hidden_dim_2 = conf.hidden_dim_2
        modules = [
            nn.LayerNorm(self.feature_dim_1, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.feature_dim_1, self.hidden_dim_1),
            nn.LeakyReLU(),
        ]
        self.input_block_1 = nn.Sequential(*modules)
        modules = [
            nn.LayerNorm(self.feature_dim_2, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.feature_dim_2, self.hidden_dim_2),
            nn.LeakyReLU(),
        ]
        self.input_block_2 = nn.Sequential(*modules)
        modules = [
            nn.LayerNorm(self.hidden_dim_2 + self.hidden_dim_1, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.hidden_dim_2 + self.hidden_dim_1, self.hidden_dim),
            nn.LeakyReLU(),
        ]
        self.input_block_3 = nn.Sequential(*modules)
        self.params = nn.ModuleDict(
            {
                "encoder": nn.ModuleList(
                    [self.input_block_1, self.input_block_2, self.input_block_3]
                ),
                "classifier": nn.ModuleList(
                    [
                        self.RNA_head,
                        self.DNA_head,
                        self.MN_head,
                        self.MG_head,
                        self.CA_head,
                        self.ZN_head,
                    ]
                ),
            }
        )

    def forward(self, feat_a, feat_b):
        feat_a = self.add_noise(feat_a)
        feat_b = self.add_noise(feat_b)
        output_1 = self.input_block_1(feat_a)
        output_2 = self.input_block_2(feat_b)
        output = self.input_block_3(torch.cat((output_1, output_2), dim=-1))
        logits = self.get_logits(output)

        return logits


class LMetalSiteMultiModal(LMetalSiteBase):
    def __init__(self, conf, training=True):
        super(LMetalSiteMultiModal, self).__init__(conf, training=training)

        self.encoding_module = MULTModel(conf)

    def forward(self, feat_a, feat_b):
        feat_a = self.add_noise(feat_a)
        feat_b = self.add_noise(feat_b)
        output = self.encoding_module(feat_a, feat_b)
        logits = self.get_logits(output)

        return logits


class LMetalSiteEncoder(nn.Module):
    def __init__(self, conf, training=True):
        super(LMetalSiteEncoder, self).__init__()

        # Hyperparameters
        self.augment_eps = conf.augment_eps
        self.training = training
        self.hidden_dim_1 = conf.hidden_dim_1
        self.hidden_dim_2 = conf.hidden_dim_2
        self.feature_dim = conf.feature_dim
        modules = [
            # nn.Tanh(),
            nn.LayerNorm(self.feature_dim, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.feature_dim, self.hidden_dim_1),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim_1, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.LeakyReLU(),
        ]
        if self.feature_dim == 384:
            modules.insert(0, nn.Sigmoid())
        self.input_block = nn.Sequential(*modules)

        self.decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim_2, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.hidden_dim_2, self.hidden_dim_1),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim_1, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(self.hidden_dim_1, self.feature_dim),
            # nn.Tanh(),
        )
        self.params = nn.ModuleDict(
            {
                "encoder": self.input_block,
                "decoder": self.decoder,
            }
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_noise(self, input):
        if self.training and self.augment_eps > 0:
            input = input + self.augment_eps * torch.randn_like(input)
        return input

    def forward(self, protein_feat, mask=None):
        protein_feat = self.add_noise(protein_feat)
        h_V = self.input_block(protein_feat)
        h_V = self.decoder(h_V)

        return h_V


class LMetalSiteTransformerEncoder(LMetalSiteEncoder):
    def __init__(self, conf, training=True):
        super(LMetalSiteTransformerEncoder, self).__init__(conf, training=training)

        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     self.hidden_dim,
        #     conf.num_heads,
        #     dim_feedforward=4 * self.hidden_dim,
        #     dropout=conf.dropout,
        #     batch_first=True,
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     self.encoder_layer, conf.num_encoder_layers
        # )
        self.encoder_layers = nn.ModuleList(
            [
                TransformerLayer(conf.hidden_dim, conf.num_heads, conf.dropout)
                for _ in range(conf.num_encoder_layers)
            ]
        )
        self.params.update(
            {"encoder": nn.ModuleList([self.input_block, self.encoder_layers])}
        )

    def forward(self, protein_feat, mask=None):
        protein_feat = self.add_noise(protein_feat)
        h_V = self.input_block(protein_feat)
        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)
        # h_V = self.transformer_encoder(h_V, mask=mask)
        h_V = self.decoder(h_V)

        return h_V
