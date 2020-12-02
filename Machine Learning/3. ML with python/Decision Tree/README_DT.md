## Homework details

这里使用了决策树模型，其本身具有随机性。共运行20次，结果放在了Results.xlsx中。



观察到三个feature组合中，始终含有**miR-122**，且多种组合train_auc与test_auc都相同，可以说明miR-122占feature的主导地位。选择一个在valudation set表现最好的组合（**train_auc = 0.947, test_auc = 0.911, validation_auc = 0.825**），即**'LINC01226', 'hsa_circ_0073052', 'miR-122'**。

