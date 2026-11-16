# Dataset acquisition

The submitted experiments use public tabular datasets and construct the
long-tailed distributions independently inside each fold.

## 数据集

| Dataset | Script key | Source/acquisition | Task |
|---|---|---|---|
| Default Credit | `default_credit` | `default-of-credit-card-clients`, `default_credit_card_clients`, `default-credit-card-clients` | 二分类 |
| Adult | `adult` | `adult` | 二分类 |
| Bank Marketing | `bank_marketing` | `bank-marketing` | 二分类 |
| Credit Score Classification | `credit_score` | [Kaggle source](https://www.kaggle.com/datasets/parisrohan/credit-score-classification); place `train.csv` at `data/credit_score_classification/train.csv` | 三分类 |

## 加载方式

The first three datasets are downloaded through `sklearn.datasets.fetch_openml`:

```python
fetch_openml(name=name, version=1, as_frame=False, parser="liac-arff")
```

The first run requires network access and scikit-learn caches the payloads
locally. Credit Score Classification is not automatically redistributed by this
repository. Download it from the source cited by the paper and place it at the
path above. The multiclass entry point is:

```bash
python3 code/scripts/run_multiclass_credit_score_experiment.py \
  --data data/credit_score_classification/train.csv
```

## 长尾构造

实验中使用以下少数类比例：

```text
0.03, 0.02, 0.01
```

每个 fold 内，训练集和测试集都会按目标少数类比例重采样。少数类标签由该 fold 的训练集类别计数自动确定。

## VFL 特征划分

本实验采用 VertiBench-style 异质性划分：

```text
balanced
importance_skew
```

每个数据集被划分为 3 个 party，其中 party 0 为 active party。

## 复现实验设置

```text
datasets = default_credit, adult, bank_marketing
ratios = 0.03, 0.02, 0.01
heterogeneity = balanced, importance_skew
splits = 3
max_samples = 6000
random_state = 13
```
