import torch.nn as nn
from .model import LMetalSite


class MetalIonSiteClassification(nn.Module):
    def __init__(self, backbone, config, ion_type="ZN"):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.ion_type = ion_type
        
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
            self.ion_type,
        )

    def forward(self, input_ids, attention_mask):
        embedding = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # replace min_max normalization?
        embedding = self.sigmoid(embedding.last_hidden_state)
        outputs = self.classifier(embedding, attention_mask)
        return outputs
