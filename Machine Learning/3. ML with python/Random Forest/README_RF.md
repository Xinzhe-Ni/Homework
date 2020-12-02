## Homework details

这里使用了随机森林模型，其本身具有随机性。共运行20次，结果放在了Results.xlsx中。



观察到三个feature组合中，始终含有**miR-122**，且多种组合train_auc与test_auc都相同，可以说明miR-122占feature的主导地位。选择一个在valudation set表现最好的组合（**train_auc = 1, test_auc = 0.983, validation_auc = 0.95**），即**'LINC01226',  'miR-122', 'miR192'**。

