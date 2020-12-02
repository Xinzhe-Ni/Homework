library("ROCR")

#读取数据
data_mat <- read.csv("/Users/xinzhe/Desktop/Lulab/毕设/gitbook/BreastCancer.csv", header = T)
for(i in 1:length(data_mat[,7])){
  if(is.na(data_mat[i,7]))
    data_mat[i,7] <- 0
}
rm(i)

#数据归一化
X <- rep(0,9*699)
dim(X) <- c(699, 9)
for(i in 2:(dim(data_mat)[2]-1)){
  X[,i-1] <- (data_mat[, i] - min(data_mat[, i])) / (max(data_mat[, i])-min(data_mat[, i]))
}
rm(i)

#数据切分
label_mat <- as.character(data_mat[, 11])
X <- data.frame(X, label_mat)
set.seed(520)
train <- sample(nrow(X), 0.75*nrow(X))
X_train <- X[train,]
Y_train <- as.character(X_train[, 10])
X_test <- X[-train,]
Y_test <- as.character(X_test[, 10])

#模型训练测试
logistic <- glm(label_mat ~ ., data = X_train, family = binomial())

#预测
predicted <- as.numeric(predict(logistic, as.data.frame(X_test), type = 'response') > 0.5)
pred <- prediction(predicted, Y_test)
auc <- performance(pred, 'auc')@y.values

# 输出AUC
auc

#画ROC曲线
roc <- performance(pred, "tpr", "fpr")
plot(roc, main = 'ROC Curve', col = "blue", lwd = 2)
abline(a = 0, b = 1, lwd = 2, col="grey", lty = 2)
legend("bottomright", inset=0.01,c("ROC", "Random Chance"), col=c("blue", "grey"), lty = c(1, 2))
