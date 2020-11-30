## Homework details

### 1. Meanings of DaPars_Test_data_All_Prediction_Results.txt columns

Gene：发生APA的基因转录本编号；



fit_value：拟合值，越大说明拟合越好；



Predicted_Proximal_APA：预测的近端APA位点；



Loci：基因在染色体上的位置；



A_1_long_exp：在组A_1中，在远端APA的表达值；



A_1_short_exp：在组A_1中，在近端APA的表达值；



A_1_PDUI：PDUI指的是Percentage of Distal polyA site Usage Index，即远端多聚位点使用指数百分比，可以衡量3‘ UTR的缩短或延长，进而判断APA位点的偏好；



Group_A_Mean_PDUI：组A的平均PDUI，这里组A只有一个样本，因此与A_1_PDUI相同；



PDUI_Group_diff：组A与组B的差异PDUI水平；



P_val：P值；



adjusted.P_val：Q值；



Pass_Filter：是否通过预设的cutoff值。



### 2. Filter

pass_filter1.txt：按docker中的cutoff值，通过的基因转录本；



pass_filter2.txt：按作业中的cutoff值，通过的基因转录本。



第一个脚本过滤出12个基因转录本，第二个脚本也过滤出12个基因转录本。由于筛选标准相同，因此可反过来验证脚本的正确性。