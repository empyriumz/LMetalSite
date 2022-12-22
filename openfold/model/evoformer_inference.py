# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn

from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.utils.feats import build_extra_msa_feat
from openfold.model.embedders import (
    InputEmbedder,
    ExtraMSAEmbedder,
)

class Evoformer(nn.Module):
    """
    Evoformer
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(Evoformer, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.extra_msa_config = self.config.extra_msa
        
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        if self.extra_msa_config.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )
        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        

    def forward(self, feats):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        n_seq = feats["msa_feat"].shape[-3]
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize the MSA and pair representations
        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

        # if self.config.extra_msa.enabled:
        #     # [*, S_e, N, C_e]
        #     a = self.extra_msa_embedder(build_extra_msa_feat(feats))
        #     z = self.extra_msa_stack(
        #             a,
        #             z,
        #             msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
        #             chunk_size=self.globals.chunk_size,
        #             use_lma=self.globals.use_lma,
        #             pair_mask=pair_mask.to(dtype=m.dtype),
        #             inplace_safe=inplace_safe,
        #             _mask_trans=self.config._mask_trans,
        #         )
        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        return outputs