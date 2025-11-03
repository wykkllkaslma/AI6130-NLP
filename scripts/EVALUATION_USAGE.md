# MedRAG 评估使用说明

## 评估脚本功能

评估 MedRAG 系统在医疗对话数据集上的性能,包括:
- **Precision**: 模型推荐的药物中,有多少是真实推荐的
- **Recall**: 真实推荐的药物中,有多少被模型成功推荐
- **F1 Score**: Precision 和 Recall 的调和平均数
- **BERTScore**: 回答的语义相似度

## 快速开始

### 1. 准备数据
确保有以下文件:
- 评估数据集: `data/normalized/meddialog_en/eval_sample_1000.json`
- 药物列表: `data/drug_list_en_expanded.txt`

### 2. 运行评估

#### 评估前 30 条数据 (快速测试)
```bash
cd /home/msai/{你们的用户名}/RAG

PYTHONPATH=/home/msai/{你们的用户名}/RAG python3 scripts/evaluate_medrag.py \
  --dataset data/normalized/meddialog_en/eval_sample_1000.json \
  --drug-list data/drug_list_en_expanded.txt \
  --limit 30 \
  --output eval_30_results.json
```

#### 评估完整 1000 条数据
```bash
cd /home/msai/{你们的用户名}/RAG

PYTHONPATH=/home/msai/{你们的用户名}/RAG python3 scripts/evaluate_medrag.py \
  --dataset data/normalized/meddialog_en/eval_sample_1000.json \
  --drug-list data/drug_list_en_expanded.txt \
  --output eval_1000_results.json
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | ✓ | - | 对话数据集文件路径 (JSON 格式) |
| `--drug-list` | ✓ | - | 药物列表文件路径 (每行一个药物名) |
| `--limit` | ✗ | None | 限制评估样本数量,用于快速测试 |
| `--output` | ✗ | evaluation_results.json | 输出结果文件路径 |

## 输出格式

评估完成后,会生成 JSON 格式的结果文件:

```json
{
  "dataset": "data/normalized/meddialog_en/eval_sample_1000.json",
  "n_examples": 30,
  "drug_metrics": {
    "precision": 0.15,
    "recall": 0.42,
    "f1": 0.22
  },
  "bertscore": {
    "precision": 0.160,
    "recall": 0.232,
    "f1": 0.194
  }
}
```

### 指标说明

#### 药物推荐指标 (drug_metrics)
- **Precision**: 模型推荐的药物中,有多少确实是医生推荐的
  - 计算公式: |推荐药物 ∩ 真实药物| / |推荐药物|
  - 值越高表示推荐越准确
  
- **Recall**: 医生推荐的药物中,有多少被模型成功推荐出来
  - 计算公式: |推荐药物 ∩ 真实药物| / |真实药物|
  - 值越高表示覆盖越全面
  
- **F1**: Precision 和 Recall 的调和平均数
  - 计算公式: 2 * (Precision * Recall) / (Precision + Recall)
  - 综合衡量推荐质量

#### 语义相似度指标 (bertscore)
- 衡量模型回答与参考回答的语义相似度
- 值越接近 1.0 表示回答质量越高

## 评估过程

1. **加载数据**: 读取对话数据集和药物列表
2. **逐条评估**: 
   - 格式化患者对话为查询
   - 调用 RAG 系统获取回答和检索文献
   - 从回答中提取药物名称
3. **计算指标**: 
   - Precision: 推荐药物中命中真实药物的比例
   - Recall: 真实药物中被推荐的比例
   - F1: Precision 和 Recall 的调和平均数
   - BERTScore: 回答与参考回答的语义相似度
4. **保存结果**: 输出到指定 JSON 文件

## 注意事项

- 前 3 个样本会显示详细的对话历史和评估信息
- 其余样本只显示进度条,不输出详细信息
- 评估需要调用 Deepseek API,请确保 API 密钥已配置
- 大数据集评估可能需要较长时间,建议先用 `--limit` 测试

## 常见问题

### Q: 如何加快评估速度?
A: 使用 `--limit` 参数限制样本数量,例如 `--limit 100`

### Q: 评估结果如何解读?
A: 
- Precision 接近 1.0 表示推荐准确(推荐的基本都对)
- Recall 接近 1.0 表示覆盖全面(该推荐的都推荐了)
- F1 综合衡量,越高越好
- BERTScore F1 接近 1.0 表示回答质量高

### Q: 为什么有些样本被跳过?
A: 样本必须同时包含患者查询和真实药物才会被评估
