## Homework details

### 1. Structure-seq

Structure-seq分为多种，是为了鉴定RNA二级结构发展而来的测序手段。其中，Shape-seq和Shape-MaP都用到了2’‐hydroxyl acylation（2’-羟基酰化）的原理。由于RNA二级结构的有无，核苷酸分为配对和非配对两种状态。



Shape-seq先将独特的编码添加到RNA 3‘端，之后在体外折叠，折叠后的RNA用SHAPE试剂1M7（1-甲基-7-亚硝基尿酸酐）处理，中断RT反转录过程，最后通过深度测序来得到1M7占据位置的核苷酸信息。



Shape-MaP原理类似，RNA折叠之后，利用SHAPE试剂1M7标记2‘-OH基团，之后在RT期间通过MaP诱导产生非互补核苷酸突变，建库、测序、比对，根据突变位点最终得到二级结构的区域。



### 2. Reactivity

Reactivity是反映Structure-seq中，RNA二级结构的指标。在mutation counting的过程中，由于Shape会引入突变，所以在基因组长度上，每个位点上所有reads的mutation的counts就构成了这个位点的Reactivity。Reactivity越大，说明二级结构的可能性越高，用于Shape profile中。