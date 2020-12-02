import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#读取数据
data_mat = []
label_mat = []
with open('/Users/xinzhe/Desktop/Lulab/毕设/gitbook/BreastCancer.csv', 'r') as f:
    for line in f.readlines()[1:]:
        line_arr = line.strip().split(',')
        if line_arr[6] == 'NA':
            line_arr[6] = 0
        data_mat.append([float(line_arr[1]), float(line_arr[2]), float(line_arr[3]), float(line_arr[4]),
        float(line_arr[5]), float(line_arr[6]), float(line_arr[7]), float(line_arr[8]), float(line_arr[9])])
        label_mat.append(line_arr[10])

#数据归一化
scaler = MinMaxScaler()
X = scaler.fit_transform(data_mat)

#PCA
transform = PCA(2)
X_pca = transform.fit_transform(X)
X_, Y_ = X_pca, label_mat

#PCA可视化
pca_data = pd.DataFrame(X_)
pca_data.columns = ['PCA_1', 'PCA_2']
pca_data['label'] = np.array(Y_)
g = sns.scatterplot(x = "PCA_1", y = "PCA_2", hue = "label", data = pca_data, s = 50)
plt.legend(loc = 'best')
plt.show()
plt.close()

#tSNE
transform = TSNE(2)
X_tsne = transform.fit_transform(X)
X_, Y_ = X_tsne, label_mat

##tSNE可视化
tSNE_data = pd.DataFrame(X_)
tSNE_data.columns = ['tSNE_1', 'tSNE_2']
tSNE_data['label'] = np.array(Y_)
g = sns.scatterplot(x = "tSNE_1", y = "tSNE_2", hue = "label", data = tSNE_data, s = 50)
plt.legend(loc = 'best')
plt.show()
plt.close()
