import torch.nn as nn
from .model import LMetalSite
import torch


class MetalIonSiteClassification(nn.Module):
    def __init__(self, backbone, config, training=True, ligand="ZN"):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.ligand = ligand
        self.training = training

        if self.config.training.fix_backbone_weight:
            self.backbone.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()
        self.classifier = LMetalSite(
            self.config.model.feature_dim,
            self.config.model.hidden_dim,
            self.config.model.num_encoder_layers,
            self.config.model.num_heads,
            self.config.model.augment_eps,
            self.config.model.dropout,
            training=self.training,
            ligand=self.ligand,
        )

    def forward(self, input_ids, attention_mask):
        embedding = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # replace min_max normalization?
        representation = self.sigmoid(embedding.last_hidden_state)
        del embedding
        self.classifier.training = self.training
        outputs = self.classifier(representation, attention_mask)
        return outputs


class MetalIonSiteEvoformer(nn.Module):
    def __init__(self, backbone, config, ligand="ZN"):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.ligand = ligand

        if self.config.training.fix_backbone_weight:
            self.backbone.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()
        self.classifier = LMetalSite(
            self.config.model.feature_dim,
            self.config.model.hidden_dim,
            self.config.model.num_encoder_layers,
            self.config.model.num_heads,
            self.config.model.augment_eps,
            self.config.model.dropout,
            self.ligand,
        )

    def forward(self, input_feature):
        embedding = self.backbone(input_feature)
        # replace min_max normalization?
        representation = self.sigmoid(embedding["single"])
        mask = torch.ones(input_feature["aatype"])
        outputs = self.classifier(representation, mask)
        return outputs
