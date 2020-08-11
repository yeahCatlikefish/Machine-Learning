import pandas as pd
import numpy as np


# 导入数据集
data = pd.read_csv("play.csv")
# 显示数据集
print(data)

# 可以看出这是一个分类变量的数据集。然后，我们就要将它变成数值变量，好利于下面的建模。
from sklearn.preprocessing import LabelEncoder
print(data.columns)
# 数据预处理
# sklearn要求数据输入的特征值（属性）features以及输出的类，必须是数值型的值，而不能是类别值（如outlook属性中的high、overcast、rainy）
# 使用LabelEncoder对特征进行硬编码(编码为0~n-1(n为种类数))
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])  # 对数据先拟合在标准化
# 打印显示对应的编码
print(data)
print('\n')

# 开始提取训练集与测试集
# 导入训练集和测试集切分包
from sklearn.model_selection import train_test_split
y = data['play']
X = data.drop('play', axis=1)
# 将数据进行分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2, train_size=0.8)
'''
X:样本特征集
y:样本标签集
random_state：是随机数的种子。在同一份数据集上，相同的种子产生相同的结果，不同的种子产生不同的划分结果
test_size：样本占比，测试集占数据集的比重，如果是整数的话就是样本的数量
x_train,y_train:构成了训练集
x_test,y_test：构成了测试集
'''
columns = X_train.columns




# 接着标准化训练集
# 数据标准化 保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train.astype(float))
X_test = ss_X.transform(X_test.astype(float))


# 构建决策树模型
from sklearn.tree import DecisionTreeClassifier
# 初始化树模型
model_tree = DecisionTreeClassifier()  # ()里面不填任何属性默认为Gini，里面填写criterion='entropy'使用的是信息熵
# 拟合数据集
model_tree.fit(X_train, y_train)


# 评价模型准确性:使用决策树对测试数据进行类别预测
y_prob = model_tree.predict_proba(X_test)[:,1]
# np.where(condition, x, y)；满足条件(condition)，输出x，不满足输出y。
y_pred = np.where(y_prob > 0.5, 1, 0)
# 预测的精准度
model_tree.score(X_test, y_pred)



# 可视化树图
data_ = pd.read_csv("play.csv")
# 特征列名
data_feature_name = data_.columns[:-1]
# 标签分类
data_target_name = np.unique(data_["play"])


import pydotplus
from sklearn import tree
from IPython.display import Image
import os

# 设置环境变量：因为scikit-learn决策树结果的可视化需要使用到Graphviz
os.environ["PATH"] += os.pathsep + 'D:\Tools\graphviz\Graphviz 0.1.1\bin'
# 可视化决策树
dot_tree = tree.export_graphviz(model_tree,feature_names=data_feature_name,class_names=data_target_name,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_tree)
img = Image(graph.create_png())
# 输出图片
graph.write_png("out_01.png")
