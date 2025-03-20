import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class cWDLDataset(Dataset):
    def __init__(self, df, data, mask_sizes, sparse_features, dense_features, target):
        super().__init__()
        self.df = df
        self.data = data
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.target = target
        self.df.index = pd.to_datetime(self.df.index)
        self.mask_sizes = mask_sizes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = 0
        # 初始化 mask
        mask = np.zeros(self.mask_sizes[0])

        # 获取当前行的标签和特征
        y_true = self.df.iloc[index][self.target]
        X_train = self.df.drop(self.target, axis=1).iloc[index]

        # 获取当前交易的时间戳作为索引
        current_date = self.df.index[index]

        # 计算三个月前的日期
        three_months_ago = current_date - pd.DateOffset(months=3)


        # 筛选出当前交易时间之前 3 个月内的所有交易记录
        prior_transactions = self.data[
            (self.data.index < current_date) &
            (self.data.index >= three_months_ago)
            ]


        nearby_found = 0  # 记录找到的附近交易数量

        context_x = np.zeros((self.mask_sizes[0], len(self.sparse_features) + len(self.dense_features)))
        context_y = np.zeros(self.mask_sizes[0])

        if not prior_transactions.empty:
            # 找到经纬度附近的数据
            nearby_transactions = prior_transactions[
                ((prior_transactions['longitude'] - X_train['longitude']) ** 2 +
                 (prior_transactions['latitude'] - X_train['latitude']) ** 2) ** 0.5 < 0.01  # 调整距离
            ]

            # 检查附近的交易记录
            for _, transaction in nearby_transactions.iterrows():
                if nearby_found >= self.mask_sizes[0]:  # 不超过 mask_sizes[0]
                    break

                mask[nearby_found] = 1  # 将 mask 中相应位置的值置为 1
                context_x[nearby_found] = transaction[self.dense_features + self.sparse_features].values  # 加入特征
                context_y[nearby_found] = transaction[self.target].values  # 加入对应的 target
                nearby_found += 1

                    # 如果找到的记录不足 mask_sizes[0]，用当前交易数据来补充
        while nearby_found < self.mask_sizes[0]:
            context_x[nearby_found] = X_train[self.dense_features + self.sparse_features].values
            context_y[nearby_found] = 0
            nearby_found += 1

            # 将数据转换为 tensor
        context_x_tensor = torch.tensor(context_x, dtype=torch.float32)
        context_y_tensor = torch.tensor(context_y, dtype=torch.float32).view(-1, 1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        y_true_tensor = torch.tensor(y_true.values, dtype=torch.float32)
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

        if context_x_tensor.shape[0] != self.mask_sizes[0]:
            print('error')
        #print(nearby_found)
        return context_x_tensor, context_y_tensor, mask_tensor, X_train_tensor, y_true_tensor  # 也返回 mask

if __name__ == '__main__':
    print('end')
