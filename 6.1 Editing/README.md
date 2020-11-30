## Homework details

### 1. Meanings of chr1.editingSites.vcf and chr1.editingSites.gvf

chr1.editingSites.vcf：包含所有编辑位点的vcf文件，内容包括位点位置、碱基变异类型以及一些信息等，其中比较有用的信息包括所在区域是exon、intron、3‘UTR或5、UTR区域；



chr1.editingSites.gvf：不光包括编辑位点，还包括了每个编辑位点的附加信息，包括基因名、变异区域、碱基变异类型、总reads数、发生编辑的reads数、编辑的比率（ratio）等。



### 2. Summary of editing sites

基于chr1.editingSites.gvf进行统计：

| exon | intron | 3'UTR | 5'UTR |
| ---- | ------ | ----- | ----- |
| 4    | 2      | 2     | 0     |

