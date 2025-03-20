import torch
from torch import nn
import torch.nn.functional as F

class Linear(nn.Module):
    """
    Linear part
    """

    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


class Dnn(nn.Module):
    """
    Dnn part
    """

    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout: 失活率
        """
        super(Dnn, self).__init__()

        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)

        x = self.dropout(x)
        return x


class WideDeep(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dense_features, sparse_features, hidden_units, dnn_dropout=0.):
        super(WideDeep, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = dense_features, sparse_features

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim)
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0,
                            (len(self.dense_feature_cols)+1) + len(self.sparse_feature_cols) * embedding_dim)
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear((len(self.dense_feature_cols)+1))
        self.final_linear = nn.Linear(hidden_units[-1], 10)

    def forward(self, x):
        dense_input, sparse_inputs = x[:, :(len(self.dense_feature_cols)+1)], x[:, (len(self.dense_feature_cols)+1):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = []
        for i in range(sparse_inputs.shape[1]):
            try:
                embed_key = 'embed_' + str(i)
                if embed_key in self.embed_layers:
                    embed_layer = self.embed_layers[embed_key]
                    sparse_embeds.append(embed_layer(sparse_inputs[:, i]))
                else:
                    print(f"Warning: {embed_key} not found in embed_layers.")
            except IndexError as e:
                print(f"IndexError at column {i}: {e}")

        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)

        # Wide
        wide_out = self.linear(dense_input)

        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)

        # out
        outputs = 0.5 * (wide_out + deep_out)

        return outputs

class DeepFM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dense_features, sparse_features,
                 hidden_units, dnn_dropout=0.):
        """
                cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
                nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况
        """
        super(DeepFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = dense_features, sparse_features
        self.cate_fea_size = len(num_embeddings)
        self.nume_fea_size = len(dense_features) + 1

        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in num_embeddings])  # 类别特征的一阶表示

        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embedding_dim) for voc_size in num_embeddings])

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + 1 + len(self.sparse_feature_cols) * embedding_dim)
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols)+1)
        self.final_linear = nn.Linear(hidden_units[-1], 10)

    def forward(self, target_x):
        dense_input, sparse_inputs = target_x[:, :(len(self.dense_feature_cols) + 1)], target_x[:,
                                                                                (len(self.dense_feature_cols) + 1):]

        sparse_inputs = sparse_inputs.long()

        """FM 一阶部分"""
        fm_1st_sparse_res = [emb(sparse_inputs[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]

        if dense_input is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(dense_input)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res  # [bs, 1]

        """FM 二阶部分"""
        fm_2nd_order_res = [emb(sparse_inputs[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, emb_size]

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)

        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)
        dnn_out = torch.cat([dnn_out, dense_input], axis=-1)

        dnn_out = self.dnn_network(dnn_out)

        dnn_out = self.final_linear(dnn_out)

        outputs = fm_1st_part + fm_2nd_part + dnn_out

        return outputs


class BiInteractionLayer(nn.Module):
    def __init__(self):
        super(BiInteractionLayer, self).__init__()

    def forward(self, embeddings):
        # 计算特征交互
        sum_of_square = torch.sum(embeddings, dim=1) ** 2
        square_of_sum = torch.sum(embeddings ** 2, dim=1)
        bi_interaction = 0.5 * (sum_of_square - square_of_sum)
        return bi_interaction

class NFM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dense_features, sparse_features,
                 hidden_units, dnn_dropout=0.):
        super(NFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = dense_features, sparse_features

        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim)
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        self.bi_interaction = BiInteractionLayer()

        input_size = embedding_dim + len(self.dense_feature_cols) + 1
        hidden_units.insert(0, input_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(dnn_dropout),
            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.ReLU(),
            nn.Dropout(dnn_dropout),
            nn.Linear(hidden_units[2], 10)
        )

    def forward(self, target_x):
        dense_input, sparse_inputs = target_x[:, :(len(self.dense_feature_cols) + 1)], target_x[:,
                                                                                    (len(self.dense_feature_cols) + 1):]
        sparse_inputs = sparse_inputs.long()

        sparse_embeds = []
        for i in range(sparse_inputs.shape[1]):
            embed_key = 'embed_' + str(i)
            if embed_key in self.embed_layers:
                sparse_embeds.append(self.embed_layers[embed_key](sparse_inputs[:, i]))

        sparse_embeds = torch.stack(sparse_embeds, dim=1)

        bi_interaction_output = self.bi_interaction(sparse_embeds)

        combined_input = torch.cat([dense_input, bi_interaction_output], axis=-1)

        outputs = self.mlp(combined_input)

        return outputs


class SENet(nn.Module):
    def __init__(self, field_size, reduction_ratio=3):
        super(SENet, self).__init__()
        reduced_size = int(max(1, field_size // reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(field_size, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, field_size),
        )

    def forward(self, embeddings):
        Z = torch.mean(embeddings, dim=-1)
        A = self.excitation(Z)
        A = A.unsqueeze(-1)
        return embeddings * A

class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, embeddings):
        sum_of_square = torch.sum(embeddings, dim=1) ** 2
        square_of_sum = torch.sum(embeddings ** 2, dim=1)
        bi_interaction = 0.5 * (sum_of_square - square_of_sum)
        return bi_interaction


class FiBiNET(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dense_features, sparse_features,
                 hidden_units, reduction_ratio=3):
        super(FiBiNET, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = dense_features, sparse_features

        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim)
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        self.senet = SENet(field_size=len(self.sparse_feature_cols), reduction_ratio=reduction_ratio)

        self.bi_pooling = BiInteractionPooling()

        input_size = len(self.dense_feature_cols) + embedding_dim + 1
        hidden_units.insert(0, input_size)
        self.dnn_network = nn.Sequential(
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units[2], 10)
        )

    def forward(self, target_x):
        dense_input, sparse_inputs = target_x[:, :(len(self.dense_feature_cols) + 1)], target_x[:,
                                                                                    (len(self.dense_feature_cols) + 1):]
        sparse_inputs = sparse_inputs.long()

        sparse_embeds = []
        for i in range(sparse_inputs.shape[1]):
            embed_key = 'embed_' + str(i)
            if embed_key in self.embed_layers:
                sparse_embeds.append(self.embed_layers[embed_key](sparse_inputs[:, i]))

        sparse_embeds = torch.stack(sparse_embeds, dim=1)

        recalibrated_embeds = self.senet(sparse_embeds)

        bi_interaction = self.bi_pooling(recalibrated_embeds)

        combined_input = torch.cat([dense_input, bi_interaction], dim=-1)

        outputs = self.dnn_network(combined_input)
        return outputs