import torch.nn as nn
from .model import LMetalSite


class MetalIonSiteClassification(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.config = config
        self.backbone = backbone

        if self.config.training.fix_backbone_weight:
            self.backbone.requires_grad_(False)

        self.classifier = LMetalSite(
            self.config.model.feature_dim,
            self.config.model.hidden_dim,
            self.config.model.num_encoder_layers,
            self.config.model.num_heads,
            self.config.model.augment_eps,
            self.config.model.dropout,
            self.config.model.ion_type,
        )

    def forward(self, input_ids, attention_mask):
        embedding = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state
        outputs = self.classifier(embedding, attention_mask)
        return outputs
