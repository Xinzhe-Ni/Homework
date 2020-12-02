## Homework details

There are 5 models below：

- Decision Tree
- SVM
- Random Forest
- Logistic Regression



综合来看，随机森林模型得到的AUC值最高，平均性能也最好，为最适合的模型。

各模型选出的最佳组合如下：



- Decison Tree: 'LINC01226', 'hsa_circ_0073052', 'miR-122'

- SVM: 'LINC01226', 'miR-26a', 'miR-122'

- Random Forest: 'LINC01226',  'miR-122', 'miR192'

- Logistic Regression: 'LINC01226', 'miR-21', 'miR-122'



可以看出，LINC01226和miR-122是十分重要的特征，其中miR-122的重要性又要

远远大于LINC01226，第三个特征四个模型预测的结果各不相同，取随机森林的结

果，最终选取的特征组合为：**'LINC01226',  'miR-122', 'miR192'**。