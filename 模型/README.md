# 模型文件说明

`fusion-seed42.joblib` 为课程实验模型，使用 614 条组合训练，岭回归正则化参数为 100，融合岭回归权重为 0.75。模型不适用于临床诊断、用药决策或未经验证的科学结论。

模型包含由 Open Problems – Single-Cell Perturbations 比赛数据计算的统计量。比赛数据使用规则和许可独立于本项目的代码许可证；请勿转发或再分发模型文件、原始数据或衍生数据，并请遵守[比赛规则](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/rules)。本提示无法限制公开仓库的下载与复制，也不能代替数据权利持有者的授权。

该模型使用全部训练数据，不能使用同一数据再评估独立测试性能。模型文件由 joblib 序列化，请只加载来源可信的文件。
