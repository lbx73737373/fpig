# Hybrid-Tower: Fine-grained Pseudo-query Interaction and Generation for Text-to-Video Retrieval

## Abstract


The Text-to-Video Retrieval (T2VR) task aims to retrieve unlabeled videos by textual queries with the same semantic meanings. Recent CLIP-based approaches have explored two frameworks: Two-Tower versus Single-Tower framework, yet the former suffers from low effectiveness, while the latter suffers from low efficiency. In this study, we explore a new Hybrid-Tower framework that can hybridize the advantages of the Two-Tower and Single-Tower framework, achieving high effectiveness and efficiency simultaneously. We propose a novel hybrid method, Fine-grained Pseudo-query Interaction and Generation for T2VR, ie, F-Pig, which includes a new pseudo-query generator designed to generate a pseudo-query for each video. This enables the video feature and the textual features of pseudo-query to interact in a fine-grained manner, similar to the Single-Tower approaches to hold high effectiveness, even before the real textual query is received. Simultaneously, Our method introduces no additional storage or computational overhead compared to the Two-Tower framework during the inference stage, thus maintaining high efficiency. Extensive experiments on five commonly used text-video retrieval benchmarks, including MSRVTT-1k, MSRVTT-3k, MSVD, VATEX and DiDeMo, demonstrate that our method achieves a significant improvement over the baseline, with an increase of 1.6% \~ 3.9% in R@1. Furthermore, our method matches the efficiency of Two-Tower models while achieving near state-of-the-art performance, highlighting the advantages of the Hybrid-Tower framework.

## Method

![Method
Diagram](./images/fk-framework-fat.drawio.jpg)



An overview of our proposed F-Pig for text-to-video retrieval: (a) the overall structure of our model. (b) A detailed illustration of our proposed pseudo-query generator. F-Pig can generate pseudo-query features from visual inputs to perform a fine-grained pseudo interaction of query-video in a Single-Tower manner ahead of the arrival of real textual queries. After the pseudo interaction, the learned video representations are retrieved by queries in an efficient Two-Tower manner. Since we can \`\`pre-fuse\'\' the pseudo query information into video representations, the effectiveness of Single-Tower methods and the efficiency of Two-Tower methods can be simultaneously assured.

## Qualitative Results

![Qualitative
Results](./images/fk-vis.jpg)


To analyze the effectiveness of our proposed Informativeness Token Selector (ITS) module, we visualize the selected patch-level tokens on the MSRVTT-1k test dataset. The first row in each example shows the patch-level tokens selected by ITS, which are then fed into the pseudo-query generator for fine-grained pseudo-query generation. The second row shows the attention maps within each frame. The corresponding patch tokens with the phrases in the query are highlighted in the same color. We can see that our selected tokens capture the fine-grained information needed to generate discriminative pseudo-queries.
