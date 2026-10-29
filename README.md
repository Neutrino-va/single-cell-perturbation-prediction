# 单细胞药物扰动响应预测

这是一个课程研究项目：用公开比赛数据预测已知药物在已知细胞类型中的基因响应统计量。模型将按药物计算的平均响应与多输出岭回归的预测加权融合，并在验证集选择融合权重和正则化参数。它是已有方法的探索性组合，不声称是新发明的基础算法。

## 模型输入与输出

- 输入：训练中出现过的细胞类型及药物名称（独热编码）。
- 输出：18,211 个基因各自的有方向差异表达统计量，定义为 `-log10(p) × sign(log-fold change)`；不是原始表达量或药物疗效。
- 局限：没有对未知药物、新细胞类型、新供体或临床应用作验证。

## 环境与数据

实验使用 Python 3.10.20，依赖版本见 `requirements.txt`。按[竞赛页面](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data)规则获取 `de_train.parquet`，放到 `数据/` 中，校验值见[数据说明](数据/数据获取说明.md)。该文件不随仓库分发；数据许可证与使用规则由数据提供方决定，仓库代码的 MIT 许可不适用于比赛数据。

```powershell
python -m pip install -r requirements.txt
python 数据审计.py 数据/de_train.parquet --out 数据审计结果.json
python 实验.py
python 补充实验.py
python 公平比较与误差分析.py
python 多细胞少样本探索.py
python 跨细胞相关性探索.py
python 药物均值岭回归融合实验.py
python 补充随机划分评估.py
```

请在仓库根目录运行。脚本会重新生成 `实验结果/` 的对应文件；按上述顺序运行会生成后续脚本需要的划分和结果。依赖版本或硬件不同可能导致数值有细微变化。开发过程见[项目迭代记录](项目迭代记录.md)。

## 本地训练和预测

仓库提供[训练好的模型权重](模型/fusion-seed42.joblib)及[参数说明](模型/fusion-seed42.json)。安装依赖后可直接预测：

```powershell
python 预测.py --cell "NK cells" --drug "Clotrimazole" --output 预测结果.csv
```

如需重新训练，请先准备比赛数据和 `实验结果/药物均值岭回归融合.csv`，再运行：

```powershell
python 训练保存模型.py
python 预测.py --cell "NK cells" --drug "Clotrimazole" --output 预测结果.csv
```

第一个命令采用已记录的常规组合留出 seed 42 参数（alpha=100、岭回归权重=0.75），用全部 614 行数据重训，并保存 `模型/fusion-seed42.joblib` 与对应参数说明。第二个命令输出指定组合的 18,211 个基因预测值。只支持模型训练时出现过的药物和细胞类型，未知类别会报错。上述重训模型使用了全部数据，不可用这些数据再称作独立测试集。joblib 文件只能从可信来源加载。

模型文件包含从比赛训练数据计算的药物均值等统计信息，适用的数据规则与许可请参见[比赛规则](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/rules)。代码的 MIT 许可不代表授予比赛数据或模型文件的使用权。

## 结果与解释

在原先的三个常规组合留出划分中，融合模型 MRRMSE 为 1.2636±0.0643，完整岭回归为 1.2952±0.0372。后续五个随机划分中，融合模型在四次的误差低于完整岭回归。各划分复用同一数据集且可能共享上游对照；设计是在观察先前结果后完成的，因此这些只是探索性结果，不是独立临床验证或竞赛排行榜分数。融合模型在验证集搜索 25 种参数组合，岭回归只搜索 5 个正则化参数，模型比较还受搜索预算差异影响。

使用的数据与任务来自 Open Problems – Single-Cell Perturbations (Kaggle, 2023)：https://www.kaggle.com/competitions/open-problems-single-cell-perturbations 。岭回归和类别编码采用 scikit-learn 的实现。项目代码以 MIT 许可证提供；第三方材料仍受其各自的许可证约束。

## 课程作业与使用声明

本项目仅为课程作业和学习研究所做的探索，结果未经独立验证，不可用于临床诊断、治疗或用药决策。请勿转发或再分发模型文件、比赛数据及其衍生数据；公开仓库无法技术上阻止下载或复制。此声明不替代数据权利持有者的授权，使用者须自行遵守比赛规则和数据许可。
