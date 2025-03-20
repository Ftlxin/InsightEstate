import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TabularCNP.dataloader import cWDLDataset
from TabularCNP.wdl import Dnn, Linear
import pandas as pd

class DeterministicEncoder(nn.Module):
    def __init__(self, sizes, mask_sizes):
        super(DeterministicEncoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, context_x, context_y, mask_matrix):

        mask_matrix_expanded = mask_matrix.unsqueeze(-1)
        context_x = context_x*mask_matrix_expanded
        context_y = context_y*mask_matrix_expanded
        encoder_input = torch.cat((context_x, context_y), dim=-1)
        batch_size, set_size, filter_size = encoder_input.shape
        x = encoder_input.view(batch_size * set_size, -1)
        for i, linear in enumerate(self.linears[:-1]):
            x = torch.relu(linear(x))
        x = self.linears[-1](x)
        x = x.view(batch_size, set_size, -1)
        representation = x.mean(dim=1)
        return representation


class DeterministicDecoder(nn.Module):
    def __init__(self, sizes, representation_size, num_embeddings, embedding_dim, dense_features, sparse_features,
                 hidden_units, dnn_dropout=0.):

        super(DeterministicDecoder, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = dense_features, sparse_features
        self.cate_fea_size = len(num_embeddings)
        self.nume_fea_size = len(dense_features)

        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size + representation_size, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in num_embeddings])  # 类别特征的一阶表示

        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embedding_dim) for voc_size in num_embeddings])

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + len(self.sparse_feature_cols) * embedding_dim +
                            representation_size)
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols) + representation_size)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, representation, target_x):
        dense_input, sparse_inputs = target_x[:, :len(self.dense_feature_cols)], target_x[:,
                                                                                 len(self.dense_feature_cols):]

        dense_input = torch.cat([dense_input, representation], axis=-1)
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
        # out
        outputs = fm_1st_part + fm_2nd_part + dnn_out
        # 分布
        # mu, log_sigma = torch.split(out, out.shape[-1] // 2, dim=-1)
        # sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)
        # dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return outputs


class CDeepFM(nn.Module):
    def __init__(self, encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings, embedding_dim, dense_features,
                 sparse_features, hidden_units=[256, 128, 64], dnn_dropout=0.):
        super(CDeepFM, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes, mask_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes, representation_size, num_embeddings, embedding_dim,
                                             dense_features, sparse_features, hidden_units, dnn_dropout)

    def forward(self, context_x, context_y, mask_matrix, target_x, target_y=None):
        representation = self._encoder(context_x, context_y, mask_matrix)
        outputs = self._decoder(representation, target_x)

        return outputs


def process_transaction_dates(df):

    date_slice = df[['listing_year', 'listing_month', 'listing_day']].rename(columns={
        'listing_year': 'year',
        'listing_month': 'month',
        'listing_day': 'day'
    })

    df['listing_date'] = pd.to_datetime(date_slice)
    if 'transaction_cycle' in df.columns:
        df['transaction_date'] = df['listing_date'] + pd.to_timedelta(df['transaction_cycle'], unit='d')
    else:
        df['transaction_date'] = df['listing_date']

    df.set_index('transaction_date', inplace=True)
    df.drop(columns=['listing_date'], inplace=True)

    return df


def CDeepFM_predict(test_data, cd_data):
    try:
        #特征
        dense_features = [
            'longitude', 'latitude', 'building_area', 'total_floors', 'listing_price', 'year_built',
            'price_adjustment', 'viewing_count',
            'followers_count', 'visitors_count', 'ladders', 'houses',
            'kitchens', 'bedrooms', 'living_rooms', 'bathrooms',
            'listing_year', 'listing_month', 'listing_day'
        ]
        sparse_features = [
            'district_name', 'business_area', 'community_name',
            'apartment_type', 'building_type', 'house_orientation',
            'renovation_status', 'building_structure',
            'heating_method', 'elevator_available', 'transaction_rights',
            'house_usage', 'house_age', 'house_ownership'
        ]
        target = ['avg_transaction']
        features = dense_features + sparse_features

        #数据处理
        test_data.insert(0, 'avg_transaction', 0)
        cd_data.drop(columns=['id'], inplace=True)

        cd_data = process_transaction_dates(cd_data)
        test_data = process_transaction_dates(test_data)

        cd_data = cd_data[target+features]
        test_data = test_data[target+features]

        #模型参数
        batch_size = 1
        d_x, d_in, d_out = 1, 34, 2
        representation_size = 10
        encoder_sizes = [d_in, 128, 128, 128, representation_size]
        decoder_sizes = [representation_size + d_x, 128, 128, 2]
        mask_sizes = [316, batch_size]
        hidden_units = [256, 128, 64]
        num_embeddings = [23, 179, 10695, 5, 5, 173, 4, 7, 1, 3, 12, 4, 3, 3]
        embedding_dim = 4
        k = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #模型初始化
        model = CDeepFM(
            encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings,
            embedding_dim, dense_features, sparse_features, hidden_units
        ).to(device)

        #数据集
        test_dataset = cWDLDataset(test_data, cd_data, mask_sizes, sparse_features, dense_features, target)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        #加载模型,更换路径
        model.load_state_dict(torch.load('E:\\postgraduate\\testDC24First\\testDC24First\\cd_cDeepFM.pth', map_location=torch.device('cpu')))

        # 执行预测
        try:
            for test_step, (context_x, context_y, mask_matrix, X_train, y_true) in enumerate(test_dataloader, 1):
                with torch.no_grad():
                    context_x, context_y, mask_matrix, X_train, y_true = (
                        context_x.to(device),
                        context_y.to(device),
                        mask_matrix.to(device),
                        X_train.to(device),
                        y_true.to(device),
                    )
                    output = model(context_x, context_y, mask_matrix, X_train)
        except Exception as e:
            raise RuntimeError(f"error: {e}")

        output = output.cpu().detach().numpy()

        return round(float(output), 4)

    except Exception as e:
        print(f"error: {e}")
        return None