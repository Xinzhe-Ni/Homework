import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score,auc,precision_recall_curve,average_precision_score,accuracy_score
from itertools import combinations

#读取数据
data_mat = []
label_mat = []
with open('/Users/xinzhe/Desktop/Lulab/毕设/gitbook/qPCR_data2.csv', 'r') as f:
    for line in f.readlines()[1:]:
        line_arr = line.strip().split(',')
        s = 0
        for i in range(1,12):
            if line_arr[i] != "NA":
                s = s + 1
            else:
                continue
        if s == 11:
            data_mat.append([float(line_arr[1]), float(line_arr[2]), float(line_arr[3]), float(line_arr[4]), float(line_arr[5]), float(line_arr[6]),
            float(line_arr[7]), float(line_arr[8]), float(line_arr[9]), float(line_arr[10]), float(line_arr[11])])
            label_mat.append(line_arr[12])
feature = ["SNORD3B", "HULC", "LINC01226", "hsa_circ_0073052", "hsa_circ_0080695", "miR-21", "miR-26a", "miR-27a", "miR-122", "miR-192", "miR-223"]
#数据归一化
scaler = MinMaxScaler()
X, Y = scaler.fit_transform(data_mat), label_mat
X = pd.DataFrame(X)
X.columns = feature

#数据划分
random_state = np.random.RandomState(1289237)
X_discovery, X_validation, Y_discovery, Y_validation = train_test_split(X, Y, test_size = 0.2, random_state = random_state)
X_discovery.index = range(len(X_discovery))
X_validation.index = range(len(X_validation))
print('number of discovery samples: {}, validation samples: {}'.format(X_discovery.shape[0], X_validation.shape[0]))

#模型函数
def clf_select(name):
    if name =='DT':
        clf = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 5, criterion = 'gini')
    elif name =='DT_cv':
        tree_para = {'max_depth': [3,5,7,9]}
        clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv = 5)
    elif name == 'SVM':
        clf = SVC(kernel = 'rbf', probability = True, C = 1)
    elif name == 'SVM_cv':
        tree_para = { 'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
        clf = GridSearchCV(SVC(kernel = 'rbf',probability = True), tree_para, cv = 5)
    elif name == 'RF':
        clf = RandomForestClassifier(n_estimators = 50, max_depth = 5)
    elif name == 'RF_cv':
        tree_para = {'n_estimators': [25, 50, 75],'max_depth': [3, 4, 5]}
        clf = GridSearchCV(RandomForestClassifier(), tree_para, cv = 5)
    elif name == 'LR':
        clf = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 100)
    elif name == 'LR_cv':
        tree_para = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
        clf = GridSearchCV(LogisticRegression(penalty = 'l2',solver = 'liblinear'), tree_para, cv = 5)
    return clf

#计算特征组合
feature_list = []
for i in combinations(feature, 3):
    feature_list.append(list(i))

#交叉验证
result_train = pd.DataFrame(columns = {'feature', 'AUC_mean'})
result_test = pd.DataFrame(columns = {'feature', 'AUC_mean'})

for j in range(len(feature_list)):
    print(j)
    skf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
    result_train_ = pd.DataFrame(columns = {'num', 'AUC'})
    result_test_ = pd.DataFrame(columns = {'num', 'AUC'})
    n = 0
    for train, test in skf.split(list(X_discovery.index), Y_discovery):
        X_train = X_discovery.loc[train, feature_list[j]]
        X_test = X_discovery.loc[test, feature_list[j]]
        Y_train = np.array(Y_discovery)[train]
        Y_test = np.array(Y_discovery)[test]

        #模型训练
        clf = clf_select('SVM')
        clf.fit(X_train, Y_train)

        #模型预测结果
        pred_proba_train = clf.predict_proba(X_train)
        fpr_train, tpr_train, thresholds = roc_curve(Y_train, pred_proba_train[:, 1], pos_label = "NC")
        roc_auc_train = auc(fpr_train, tpr_train)
        pred_proba_test = clf.predict_proba(X_test)
        fpr_test, tpr_test, thresholds = roc_curve(Y_test, pred_proba_test[:, 1], pos_label = "NC")
        roc_auc_test = auc(fpr_test, tpr_test)

        result_train_.loc[n, 'num'] = n
        result_train_.loc[n, 'AUC'] = roc_auc_train
        result_test_.loc[n, 'num'] = n
        result_test_.loc[n, 'AUC'] = roc_auc_test
        n = n + 1

    #模型平均AUC计算
    result_train.loc[j, 'feature'] = ','.join(feature_list[j])
    result_train.loc[j, 'AUC_mean'] = result_train_['AUC'].mean()
    result_test.loc[j, 'feature'] = ','.join(feature_list[j])
    result_test.loc[j, 'AUC_mean'] = result_test_['AUC'].mean()

#最佳特征组合
best_feature = result_test.loc[result_test.sort_values('AUC_mean', ascending = False).index[0], 'feature']
for k in range(len(result_test['feature'])):
    if result_train.loc[k, 'feature'] == best_feature:
        AUC_train = result_train.loc[k, 'AUC_mean']
    if result_test.loc[k, 'feature'] == best_feature:
        AUC_test = result_test.loc[k, 'AUC_mean']
best_feature = best_feature.split(',')
print(best_feature)
print(AUC_train)
print(AUC_test)

#Validation set评估
clf = clf_select('SVM')
clf.fit(X_discovery.loc[:, best_feature], Y_discovery)
Y_predict = clf.predict(X_validation.loc[:, best_feature])
proba = clf.predict_proba(X_validation.loc[:, best_feature]).T[1]
fpr, tpr, thresholds = roc_curve(Y_validation, proba, pos_label = "NC")
roc_auc = auc(fpr, tpr)
print(roc_auc)

#Validation set的ROC曲线
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, '-', color='b', label='Validation AUC of {:.4f}'.format(roc_auc), lw=2)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Chance')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('ROC curve of test data')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='best',fontsize='small')
plt.tight_layout()
plt.show()
plt.close()
