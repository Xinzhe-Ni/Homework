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

#PCA
X_pca <- prcomp(X, center = T, scale = F, rank. = 2)

#PCA可视化
colors <- rainbow(length(unique(data_mat[,11])))
colors_plot <- as.character(data_mat[,11])
colors_plot[which(colors_plot == "benign")] <- colors[1]
colors_plot[which(colors_plot == "malignant")] <- colors[2]
plot(X_pca$x, col = colors_plot, asp = 1, pch = 20,
     xlab = "component_1", ylab = "component_2", main = "PCA plot")
legend("topright", c("benign", "malignant"), col=c(colors[1], colors[2]), lty = 1)

#tSNE
X_tsne <- Rtsne(X, dims = 2, check_duplicates = F)

#tSNE可视化
colors <- rainbow(length(unique(data_mat[,11])))
colors_plot <- as.character(data_mat[,11])
colors_plot[which(colors_plot == "benign")] <- colors[1]
colors_plot[which(colors_plot == "malignant")] <- colors[2]
plot(X_tsne$Y, col = colors_plot, asp = 1, pch = 20,
     xlab = "tSNE_1", ylab = "tSNE_2", main = "tSNE plot")
legend("topleft", c("benign", "malignant"), col=c(colors[1], colors[2]), lty = 1)
