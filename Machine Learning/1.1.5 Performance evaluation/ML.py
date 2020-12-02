import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
X, Y = scaler.fit_transform(data_mat), label_mat

#数据切分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 520, test_size = 0.25)

#模型训练测试
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

#预测
Y_predict = log_reg.predict(X_test)
proba = log_reg.predict_proba(X_test).T[1]
fpr, tpr, thresholds = roc_curve(Y_test, proba, pos_label = "malignant")
roc_auc = auc(fpr, tpr)

# 输出AUC
print(roc_auc)

#画ROC曲线
plt.plot(fpr, tpr, '-', color='b', label='ROC', lw=2)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Chance')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
plt.close()
