import torch
import torch.nn as nn
from config import FIELD_FEATURE_COUNT, MONSTER_COUNT


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=256, num_heads=4, num_layers=4, dropout=0.3):
        super().__init__()
        # num_units，包括怪物种类和场地特征种类
        # 怪物特征 + 场地特征 = 总特征数量
        self.num_units = num_units
        self.monster_count = MONSTER_COUNT
        self.field_count = FIELD_FEATURE_COUNT
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 注意力层与FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()
        # 删除归一化
        # self.norm.append(nn.LayerNorm(embed_dim))

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=dropout
                )
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 友方注意力层
            self.friend_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=dropout
                )
            )
            self.friend_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 初始化注意力层参数
            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)
            # self.norm.append(nn.LayerNorm(embed_dim))

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        # 提取TopK特征（怪物 + 场地）
        # 由于现在包含场地特征，需要增加k值以确保重要特征不被遗漏
        # k=8 可以保证包含主要怪物和所有场地特征
        k = min(4, left_counts.shape[1])  # 确保k不超过实际特征数
        left_values, left_indices = torch.topk(left_counts, k=k, dim=1)
        right_values, right_indices = torch.topk(right_counts, k=k, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)  # (B, k, 128)
        right_feat = self.unit_embed(right_indices)  # (B, k, 128)

        # 直接用模长表示战斗力
        left_feat = left_feat * left_values.unsqueeze(-1)
        right_feat = right_feat * right_values.unsqueeze(-1)

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, k) 0.1防一手可能的浮点误差
        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        for i in range(self.num_layers):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            delta_left, _ = self.friend_attentions[i](
                query=left_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)
            # 不要归一化，数值嵌入用模长表示战斗力
            # left_feat = self.norm[i](left_feat)
            # right_feat = self.norm[i](right_feat)

        # 计算 L2 模长作为最终战斗力指标
        # 这里不进行全连接，直接取几何距离
        # 由于战斗力组合往往是非线性的，非线性由注意力模块处理
        # 因此这里不要对向量进行求和再取模长，而是先取模长再求和
        L_norms = torch.norm(left_feat, p=2, dim=-1)
        R_norms = torch.norm(right_feat, p=2, dim=-1)

        # 乘上 mask 排除无效单位，并在 k 维度求和 (B,)
        # 在 dim=1 求和，保留 Batch 维度
        L = (L_norms * left_mask).sum(dim=1)
        R = (R_norms * right_mask).sum(dim=1)

        # 计算战斗力差输出概率，'L': 0, 'R': 1，R大于L时输出大于0.5
        output = torch.sigmoid(R - L)

        return output
