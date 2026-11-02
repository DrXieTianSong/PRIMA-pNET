import torch
import torch.nn as nn


class DualResidualMTMLP(nn.Module):
    def __init__(self, table_dim, bert_dim=768, proj_dim=64, n_heads=4,
                 n_bert_tokens=12, hidden1=128, hidden2=64, dropout=0.5):
        super().__init__()
        self.proj_dim = proj_dim
        self.n_bert_tokens = n_bert_tokens

        self.table_proj = nn.Sequential(
            nn.Linear(table_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bert_proj = nn.Linear(bert_dim, n_bert_tokens * proj_dim)
        self.bert_ln = nn.LayerNorm(proj_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(proj_dim)

        self.shared = nn.Sequential(
            nn.Linear(proj_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_grade = nn.Linear(hidden2, 1)
        self.head_ln = nn.Linear(hidden2, 1)

    def forward(self, table_feat, bert_feat):
        table_emb = self.table_proj(table_feat).unsqueeze(1)
        bert_tokens = self.bert_proj(bert_feat).view(-1, self.n_bert_tokens, self.proj_dim)
        bert_tokens = self.bert_ln(bert_tokens)
        attn_out, _ = self.cross_attn(table_emb, bert_tokens, bert_tokens)
        bert_pool = bert_tokens.mean(dim=1, keepdim=True)
        enriched = self.attn_ln(attn_out + table_emb + bert_pool)
        enriched = enriched.squeeze(1)
        h = self.shared(enriched)
        return self.head_grade(h), self.head_ln(h)
