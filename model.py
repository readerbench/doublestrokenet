from typing import Callable, Optional, Union
import torch
import math
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from rot_emb import RotaryEmbedding

class PositionalEncoding(nn.Module):
  """
  Implement the PE function.
  """
  def __init__(self, d_model, max_seq_len=50):
    super().__init__()
    self.d_model = d_model

    # create constant 'pe' matrix with values dependant on
    # pos and i
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
      for i in range(0, d_model, 2):
        pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
        pe[pos, i + 1] = math.cos(pos / (10000**((2 * (i + 1)) / d_model)))
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    # make embeddings relatively larger
    x = x * math.sqrt(self.d_model)
    #add constant to embedding
    seq_len = x.size(1)

    x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
    return x
  
class StrokeNet(nn.Module):

  def __init__(self,
              user_cnt,
              feat_cnt,
              key_cnt,
              key_emb_size,
              dim_ff,
              num_heads,
              num_layers,
              dropout,
              causal_att,
              use_user_emb) -> None:
    super().__init__()

    self.positional_encoding = PositionalEncoding(key_emb_size * 3, 51)

    self.keycode_embedding = nn.Embedding(key_cnt, key_emb_size)
    self.user_embedding = nn.Embedding(user_cnt, key_emb_size * 3)

    self.feat_cnt = feat_cnt
    self.hidden_dim = key_emb_size
    self.dim_ff = dim_ff
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.dropout = dropout
    self.causal_att = causal_att
    self.use_user_emb = use_user_emb

    self.feat_proj = nn.Linear(feat_cnt, key_emb_size, bias=True)
    self.feat_bn = nn.BatchNorm1d(key_emb_size)

    self.input_mlp = nn.Sequential(
        nn.Linear(key_emb_size * 3, key_emb_size * 3),
        nn.ReLU(),
        nn.Linear(key_emb_size * 3, key_emb_size * 3),
        nn.ReLU(),
    )

    self.trf_cross = nn.TransformerEncoder(nn.TransformerEncoderLayer(
        d_model=key_emb_size * 3,
        dim_feedforward=dim_ff,
        nhead=self.num_heads,
        dropout=self.dropout,
        batch_first=True,
        norm_first=True),
                                           num_layers=self.num_layers)

    self.electra_lin = nn.Linear(key_emb_size * 3, 2, bias=False)

    if self.use_user_emb:
      self.user_lin = nn.Linear(key_emb_size * 3, 2, bias=False)

  def forward(self, b0, b1, feat, mask, user, attn_mask=None):

    feat = self.feat_proj(feat)

    feat = feat.transpose(1, 2)
    feat = self.feat_bn(feat)
    feat = feat.transpose(1, 2)

    b0_emb = self.keycode_embedding(b0)
    b1_emb = self.keycode_embedding(b1)
    
    user_emb = self.user_embedding(user)

    x = torch.cat([b0_emb, b1_emb, feat], dim=-1)

    # append user embedding to the beginning of the sequence
    x = torch.cat([user_emb.unsqueeze(1), x], dim=1)

    # MLP
    x = self.input_mlp(x)

    # add positional encoding - including user embedding
    x = self.positional_encoding(x)

    if self.causal_att:
      x = self.trf_cross(src=x, mask=attn_mask, src_key_padding_mask=mask, is_causal=True)
    else:
      x = self.trf_cross(src=x, src_key_padding_mask=mask, is_causal=False)

    # remove user embedding
    user_out = x[:, 0]
    x = x[:, 1:]

    x = self.electra_lin(x)

    if self.use_user_emb:
      user_out = self.user_lin(user_out)
      return x, user_out
    else:
      return x, None


class MultiHeadRot(nn.Module):

  def __init__(self, d_model, nhead, dropout=0.0, bias=False):
    super().__init__()
    self.rotary_emb = RotaryEmbedding(dim=d_model // nhead)
    self.linear_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
    self.linear_out = nn.Linear(d_model, d_model, bias=bias)
    self.nhead = nhead
    self.d_model = d_model
    self.dropout = dropout
    self.active_dropout = 0.0

  def forward(self, x, src_mask, src_padding, is_causal):
    bsz = x.size(0)
    
    q, k, v = torch.split(self.linear_qkv(x), self.d_model, dim=-1)

    head_dim = self.d_model // self.nhead

    # split heads
    q = q.view(bsz, -1, self.nhead, head_dim).transpose(1, 2)
    k = k.view(bsz, -1, self.nhead, head_dim).transpose(1, 2)
    v = v.view(bsz, -1, self.nhead, head_dim).transpose(1, 2)

    # apply rotary embedding
    q = self.rotary_emb.rotate_queries_or_keys(q)
    k = self.rotary_emb.rotate_queries_or_keys(k)

    if is_causal:
      dot_prod = F.scaled_dot_product_attention(q, k, v, dropout_p=self.active_dropout, is_causal=is_causal)
    else:
      dot_prod = F.scaled_dot_product_attention(q, k, v, attn_mask=src_mask, dropout_p=self.active_dropout, is_causal=is_causal)
    # concat heads
    dot_prod = dot_prod.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)

    return self.linear_out(dot_prod)

  def train(self, mode=True):
    super().train(mode)
    if mode:
      self.active_dropout = self.dropout
    else:  
      self.active_dropout = 0.0
    
class TransformerEncoderRotLayer(nn.Module):

  def __init__(self,
               d_model,
               nhead,
               dim_feedforward=756,
               dropout=0.1):
    super().__init__()
    self.multihead_rot = MultiHeadRot(d_model=d_model, nhead=nhead, dropout=dropout)

    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    self.activation = nn.SiLU()

  def forward(self, x,  src_mask, src_key_padding_mask, is_causal):
    x = x + self.dropout1(self.multihead_rot(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal))
    x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))

    return x
  
class TransformerEncoderRot(nn.Module):
  
  def __init__(self,
               trf_layer,
               num_layers):
    super().__init__()
    self.layers = nn.ModuleList([trf_layer for _ in range(num_layers)])

  def forward(self, x, src_mask, src_key_padding_mask, is_causal):
    for layer in self.layers:
      x = layer(x, src_mask, src_key_padding_mask, is_causal)
    return x

class StrokeNetRot(nn.Module):

  def __init__(self,
              user_cnt,
              feat_cnt,
              key_cnt,
              key_emb_size,
              dim_ff,
              num_heads,
              num_layers,
              dropout,
              causal_att,
              use_user_emb) -> None:
    super().__init__()

    self.keycode_embedding = nn.Embedding(key_cnt, key_emb_size)
    self.user_embedding = nn.Embedding(user_cnt, key_emb_size * 3)

    self.feat_cnt = feat_cnt
    self.hidden_dim = key_emb_size
    self.dim_ff = dim_ff
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.dropout = dropout
    self.causal_att = causal_att
    self.use_user_emb = use_user_emb

    self.feat_proj = nn.Linear(feat_cnt, key_emb_size, bias=True)
    self.feat_bn = nn.BatchNorm1d(key_emb_size)
    # self.input_rff = GaussianFourierFeatureTransform(num_input_feats=4, mapping_size=key_emb_size // 2, scale=2)

    self.input_mlp = nn.Sequential(
        nn.Linear(key_emb_size * 3, key_emb_size * 3),
        nn.SiLU(),
        nn.Linear(key_emb_size * 3, key_emb_size * 3),
        nn.SiLU(),
    )

    self.trf_cross = TransformerEncoderRot(
          TransformerEncoderRotLayer(d_model=key_emb_size * 3, nhead=self.num_heads,
                              dim_feedforward=dim_ff, dropout=self.dropout),
          num_layers=self.num_layers
    )

    self.electra_lin = nn.Linear(key_emb_size * 3, 2, bias=False)

    if self.use_user_emb:
      self.user_lin = nn.Linear(key_emb_size * 3, 2, bias=False)

  def forward(self, b0, b1, feat, mask, user, attn_mask=None):

    feat = self.feat_proj(feat)
    # feat = self.input_rff(feat)

    feat = feat.transpose(1, 2)
    feat = self.feat_bn(feat)
    feat = feat.transpose(1, 2)

    b0_emb = self.keycode_embedding(b0)
    b1_emb = self.keycode_embedding(b1)
    
    user_emb = self.user_embedding(user)

    x = torch.cat([b0_emb, b1_emb, feat], dim=-1)

    # append user embedding to the beginning of the sequence
    x = torch.cat([user_emb.unsqueeze(1), x], dim=1)

    # MLP
    x = self.input_mlp(x)
    x = self.trf_cross(x, src_mask=attn_mask, src_key_padding_mask=mask, is_causal=self.causal_att)

    # remove user embedding
    user_out = x[:, 0]
    x = x[:, 1:]

    x = self.electra_lin(x)

    if self.use_user_emb:
      user_out = self.user_lin(user_out)
      return x, user_out
    else:
      return x, None