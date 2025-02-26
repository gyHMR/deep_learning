import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder



# 读取数据集
train_data=pd.read_csv('./train.csv')
test_data=pd.read_csv('./test.csv')

# 查看数据
print(train_data.shape,test_data.shape) #查看形状
# print(train_data.head()) #查看前几行
# print(train_data.dtypes) #查看数据类型
# missing_rows=train_data.isnull().any(axis=1)   #计算有缺失值的行数（缺失行较少就直接删除）
# print(missing_rows.sum(),train_data.shape[0])

# 特征工程
all_features=pd.concat((train_data.iloc[:,1:],test_data.iloc[:,1:])) #将训练集和测试集的特征合并(合并时去除id列)
all_features=all_features.iloc[:,all_features.columns!='Sold Price']  #去除掉目标列


# 对于数值形数据和obj形数据分开处理
# 数值型数据处理
numerice_features=all_features.select_dtypes(include=['float64','int64']).columns  #选出数值列
numerice_features=all_features[numerice_features]
for column in numerice_features.columns:
    numerice_features[column]=numerice_features[column].fillna(numerice_features[column].mean())  #填充空值
#标准化处理
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(numerice_features)
numerice_features = pd.DataFrame(train_data_scaled, columns=numerice_features.columns) # 将结果转换为 DataFrame
print(numerice_features)

# # 对于obj型数据
# obj_feature=all_features.select_dtypes(include=['object']).columns
# obj_feature=all_features[obj_feature]
# # 对对象型数据进行独热编码
# encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' 删除第一列以避免多重共线性
# obj_encoded = encoder.fit_transform(obj_feature)
# # 将编码后的结果转换为 DataFrame
# obj_encoded_df = pd.DataFrame(obj_encoded, columns=encoder.get_feature_names_out(obj_feature.columns))
# print(obj_encoded_df)


# 先仅仅只看数值型数据进行训练
#####################################线性回归






######################################MLP
# # 转换为张量
# n_train=train_data.shape[0]
# train_features=torch.tensor(numerice_features[:n_train,:],dtype=torch.float32)
# test_features=torch.tensor(numerice_features[n_train:,:],dtype=torch.float32)
# train_lables=torch.tensor(train_data['Sold Price'].reshape(-1,1),dtype=torch.float32)
# # 是否使用GPU训练
# if not torch.cuda.is_available():
#     print('CUDA is not available. Training on CPU ...')
# else:
#     print('CUDA is available. Training on GPU ...')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # 将特征和标签转移到设备上
# train_features=train_features.to(device)
# test_features=test_features.to(device)
# train_lables=train_lables.to(device)




# 均方误差损失函数













