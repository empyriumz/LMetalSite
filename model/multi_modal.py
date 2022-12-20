import torch
from torch import nn
from model.modules.transformer import TransformerEncoder

class MULTModel(nn.Module):
    def __init__(self, conf):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a = (
            conf.feature_dim_1,
            conf.feature_dim_2,
        )
        self.d_l, self.d_a = conf.hidden_dim, conf.hidden_dim
        self.num_heads = conf.num_heads
        self.encoder_layers = conf.num_encoder_layers
        self.dropout_prob = conf.dropout
        self.attn_mask = conf.attn_mask

        combined_dim = self.d_l + self.d_a
        output_dim = conf.hidden_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        # could be replaced by linear projections
        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False
        )

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type="la")          
        self.trans_a_with_l = self.get_network(self_type="al")
                
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type="l_mem", layers=self.encoder_layers)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=self.encoder_layers)
      
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # dropout layers
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.LeakyReLU()
        
    def get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al"]:
            embed_dim = self.d_l
        elif self_type in ["a", "la"]:
            embed_dim = self.d_a
        elif self_type == "l_mem":
            embed_dim = self.d_l
        elif self_type == "a_mem":
            embed_dim = self.d_a
       
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.encoder_layers, layers),
            dropout=self.dropout_prob,
            attn_mask=self.attn_mask,
        )

    def forward(self, x_l, x_a):
        """
        input features should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.dropout(x_l.transpose(1, 2))
        x_a = x_a.transpose(1, 2)
      
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l) # Dimension (N, d_l, L)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        
        # A --> L
        h_l_with_as = self.trans_l_with_a(
                proj_x_l, proj_x_a, proj_x_a
            )  # Dimension (L, N, d_l)
        h_ls = self.trans_l_mem(h_l_with_as)  
        last_h_l = last_hs = h_ls

        # L --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_as = self.trans_a_mem(h_a_with_ls)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as
        last_hs = torch.cat([last_h_l, last_h_a], dim=2).permute(1, 0, 2)

        # A residual block
        last_hs_proj = self.relu(self.proj1(last_hs))
        last_hs_proj = self.dropout(last_hs_proj)
        last_hs_proj = self.proj2(last_hs_proj)
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs
