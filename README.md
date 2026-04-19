# R2GenGraduation

This repository contains the source code for my graduation project. I tried to modify the "R2Gen-Mamba" model and applied it to road damage datasets for automatic report generation.

## Requirements

定稿之后补充。。。

## R2GenGraduation Models

The pre-trained R2GenGraduation models on different datasets. You can download the models and run them on the corresponding datasets to replicate my results.

到时候放在release里面

## Datasets

数据来源于学校实验室，无法公开但后续会补充部分样例，用类似结构的数据都可以跑通

## Train

Run `bash train？？.sh` to train a model on the 。。 data.

## Test

Run `bash test_？？.sh` to test a model on the 。。 data.

Follow or [CheXbert](https://github.com/stanfordmlgroup/CheXbert) to extract the labels and then run `python compute_ce.py`. Note that there are several steps that might accumulate the errors for the computation, e.g., the labelling error and the label conversion. We refer the readers to those new metrics, e.g., [RadGraph](https://github.com/jbdel/rrg_emnlp) and [RadCliQ](https://github.com/rajpurkarlab/CXR-Report-Metric).

## Acknowledgements

This project is developed based on [R2Gen](https://github.com/zhjohnchan/R2Gen), and we appreciate their original work and open-source contribution.
