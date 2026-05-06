import torch
import torch.nn as nn
from config import FIELD_FEATURE_COUNT, MONSTER_COUNT


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=256, num_heads=4, num_layers=3, dropout=0.3):
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
        # self.norm = nn.ModuleList()

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
            # 删除归一化
            # self.norm.append(nn.LayerNorm(embed_dim))

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        # 提取TopK特征（怪物 + 场地）
        # 由于现在包含场地特征，需要增加k值以确保重要特征不被遗漏
        # k=4 可以保证包含主要怪物和所有场地特征
        k = min(4, left_counts.shape[1])  # 确保k不超过实际特征数
        left_values, left_indices = torch.topk(left_counts, k=k, dim=1)
        right_values, right_indices = torch.topk(right_counts, k=k, dim=1)

        # 嵌入层，base 保留单体的原始特征
        # AI 可解释性结果表明，怪物单体的特征也是被需要的
        # 否则需要学习单体/数量
        left_base = self.unit_embed(left_indices)  # (B, k, 128)
        right_base = self.unit_embed(right_indices)  # (B, k, 128)

        # 直接用模长表示战斗力
        left_feat = left_base * left_values.unsqueeze(-1)
        right_feat = right_base * right_values.unsqueeze(-1)

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, k) 0.1防一手可能的浮点误差
        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        # 动态获取 Batch Size，供后续 unflatten 使用
        B = left_feat.size(0)

        for i in range(self.num_layers):
            # 敌方注意力
            # 2x2=4 组动态交互 (实验表明这 2×2 组效果最佳，而非全量的 4×2 组)
            # 利用批处理，合并左右之间的交互，从而加速运算(对旧代码的 1×2 组feat看feat也有效)
            q_enemy = torch.cat([
                left_feat, left_base,  # 左方看的 2 组: q=feat(看base), q=base(看feat)
                right_feat, right_base  # 右方看的 2 组
            ], dim=0)
            k_enemy = torch.cat([
                right_base, right_feat,  # 供左方看的 2 组目标
                left_base, left_feat  # 供右方看的 2 组目标
            ], dim=0)

            # musk 重复 2 次
            mask_enemy = torch.cat([right_mask.repeat(2, 1), left_mask.repeat(2, 1)], dim=0)

            out_enemy, _ = self.enemy_attentions[i](
                query=q_enemy, key=k_enemy, value=k_enemy,
                key_padding_mask=~mask_enemy, need_weights=False,
            )

            # 解开成 (4, B, k, d)，切片求和
            out_enemy = out_enemy.unflatten(0, (4, B))
            # x = x + attn(q_base,k_feat,v_feat) + attn(q_feat,k_base,v_base)
            left_feat = left_feat + out_enemy[:2].sum(dim=0)
            right_feat = right_feat + out_enemy[2:].sum(dim=0)

            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            q_friend = torch.cat([
                left_feat, left_base,
                right_feat, right_base
            ], dim=0)
            k_friend = torch.cat([
                left_base, left_feat,
                right_base, right_feat
            ], dim=0)

            mask_friend = torch.cat([left_mask.repeat(2, 1), right_mask.repeat(2, 1)], dim=0)

            out_friend, _ = self.friend_attentions[i](
                query=q_friend, key=k_friend, value=k_friend,
                key_padding_mask=~mask_friend, need_weights=False,
            )

            out_friend = out_friend.unflatten(0, (4, B))
            left_feat = left_feat + out_friend[:2].sum(dim=0)
            right_feat = right_feat + out_friend[2:].sum(dim=0)

            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)
            # 不要归一化，数值嵌入本质是用模长表示战斗力
            # left_feat = self.norm[i](left_feat)
            # right_feat = self.norm[i](right_feat)

        # 计算 L2 范数作为最终战斗力指标
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
