#!/bin/bash

grep -w 'Y' DaPars_Test_data_All_Prediction_Results.txt > pass_filter1.txt

cat DaPars_Test_data_All_Prediction_Results.txt | awk '$15 <= 0.05{print $0}' | awk '$13 >= 0.5{print $0}' \
| awk '$11 >= 0.59*$12{print $0}' > pass_filter2.txt

exit 0
