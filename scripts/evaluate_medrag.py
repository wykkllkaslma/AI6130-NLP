"""
评估 MedRAG 在医疗对话数据集上的表现

评估指标：
- Precision: 模型推荐的药物中,有多少是真实推荐的
- Recall: 真实推荐的药物中,有多少被模型成功推荐  
- F1 Score: Precision 和 Recall 的调和平均数
- BERTScore: 回答的语义相似度

用法示例：
    # 评估前30条数据
    python3 scripts/evaluate_medrag.py \
        --dataset data/normalized/meddialog_en/eval_sample_1000.json \
        --drug-list data/drug_list_en_expanded.txt \
        --limit 30 \
        --output results.json
    
    # 评估完整数据集
    python3 scripts/evaluate_medrag.py \
        --dataset data/normalized/meddialog_en/eval_sample_1000.json \
        --drug-list data/drug_list_en_expanded.txt \
        --output results.json

依赖：
    pip install tqdm bert-score
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from statistics import mean

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def simple_drug_extractor(text, drug_list_set=None):
    """提取文本中提到的药物名称
    
    使用最长匹配方式在 drug_list_set 中查找药物名。
    """
    if not text or not text.strip() or not drug_list_set:
        return []
    
    found = []
    text_lower = text.lower()
    # 按长度排序药物名，优先匹配较长的名称
    for drug in sorted(drug_list_set, key=lambda x: -len(x)):
        if drug in text_lower:
            found.append(drug)
    return found


def precision(preds, refs):
    """
    计算 Precision: 模型推荐的药物中,有多少是真实推荐的
    
    Precision = |推荐药物 ∩ 真实药物| / |推荐药物|
    
    Args:
        preds: 预测的药物列表的列表 [[drug1, drug2, ...], ...]
        refs: 真实的药物列表的列表 [[drug1, drug2, ...], ...]
    
    Returns:
        平均 Precision 值
    """
    scores = []
    for p, r in zip(preds, refs):
        pred_set = set([x.lower() for x in p])
        ref_set = set([x.lower() for x in r])
        
        if not pred_set:  # 如果没有推荐任何药物
            scores.append(0.0)
        else:
            correct = len(pred_set & ref_set)  # 交集
            scores.append(correct / float(len(pred_set)))
    
    return mean(scores) if scores else 0.0


def recall(preds, refs):
    """
    计算 Recall: 真实推荐的药物中,有多少被模型成功推荐
    
    Recall = |推荐药物 ∩ 真实药物| / |真实药物|
    
    Args:
        preds: 预测的药物列表的列表 [[drug1, drug2, ...], ...]
        refs: 真实的药物列表的列表 [[drug1, drug2, ...], ...]
    
    Returns:
        平均 Recall 值
    """
    scores = []
    for p, r in zip(preds, refs):
        pred_set = set([x.lower() for x in p])
        ref_set = set([x.lower() for x in r])
        
        if not ref_set:  # 如果没有真实药物
            scores.append(0.0)
        else:
            correct = len(pred_set & ref_set)  # 交集
            scores.append(correct / float(len(ref_set)))
    
    return mean(scores) if scores else 0.0


def f1_score(preds, refs):
    """
    计算 F1 Score: Precision 和 Recall 的调和平均数
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision(preds, refs)
    rec = recall(preds, refs)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def compute_bertscore(preds, refs, lang="en"):
    """计算 BERTScore"""
    if bert_score is None:
        print("Missing bert-score package. Install with: pip install bert-score")
        return {"precision":0.0, "recall":0.0, "f1":0.0}
    
    if not preds or not refs:
        print("Warning: Empty predictions or references for BERTScore")
        return {"precision":0.0, "recall":0.0, "f1":0.0}
    
    # 使用英文BERT模型
    model_type = "bert-base-uncased"
    P, R, F = bert_score(preds, refs, lang=lang, model_type=model_type, rescale_with_baseline=True)
    return {"precision": float(P.mean().item()), "recall": float(R.mean().item()), "f1": float(F.mean().item())}


def format_dialogue(dialogue):
    """Format DialMed dialogue into a single query string."""
    # Combine all patient utterances into a single query
    query_parts = []
    for turn in dialogue:
        if turn.get("speaker") == "patient":
            query_parts.append(turn.get("utterance", ""))
    return " ".join(query_parts)

def get_ground_truth_meds(example):
    """Extract ground truth medications from a DialMed example."""
    # Prefer ground_truth_drugs (added by filter script) over medications
    if "ground_truth_drugs" in example and example["ground_truth_drugs"]:
        return example["ground_truth_drugs"]
    # Fallback to medications field
    return example.get("medications", [])


def main():
    parser = argparse.ArgumentParser(description="评估 MedRAG 系统性能")
    parser.add_argument("--dataset", required=True, help="对话数据集文件路径 (JSON)")
    parser.add_argument("--drug-list", required=True, help="药物列表文件路径")
    parser.add_argument("--limit", type=int, default=None, help="限制评估样本数量 (可选)")
    parser.add_argument("--output", default="evaluation_results.json", help="输出结果文件路径")
    args = parser.parse_args()

    # Load drug list
    try:
        with open(args.drug_list, 'r') as f:
            drug_set = set([line.strip().lower() for line in f if line.strip()])
        print(f"Loaded {len(drug_set)} drug names from {args.drug_list}")
    except Exception as e:
        print(f"Failed to load drug list: {e}")
        sys.exit(1)

    print("Loading DialMed dataset from", args.dataset)
    try:
        with open(args.dataset) as f:
            ds = json.load(f)
        print(f"Loaded {len(ds)} dialogues")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    if args.limit:
        ds = ds[:args.limit]
        print(f"Using first {args.limit} examples")

    # Load answer module
    print("Loading answer module...")
    try:
        from scripts.answer_module import answer as model_answer
    except Exception as e:
        print(f"Error loading answer module: {e}")
        sys.exit(1)

    preds_texts = []    # 模型生成的回答文本
    refs_texts = []     # 参考回答文本
    preds_drugs = []    # 模型推荐的药品列表
    refs_drugs = []     # 真实药品列表

    print(f"Evaluating {len(ds)} dialogues...")
    for i, ex in enumerate(tqdm(ds), 1):
        # 只在前3个样本显示对话历史
        if i <= 3:
            print(f"\n========== 示例 {i} ==========")
            print("\n对话历史:")
            for turn in ex.get("dialogue", []):
                speaker = "患者" if turn["speaker"] == "patient" else "医生"
                print(f"{speaker}: {turn['utterance']}")
            
        # 获取查询和真实药品
        query = format_dialogue(ex.get("dialogue", []))
        if not query:
            continue
            
        true_meds = get_ground_truth_meds(ex)
        if not true_meds:
            continue
            
        # 获取模型回答
        try:
            ans, refs = model_answer(query)
            
            # 提取药物
            pred_meds = simple_drug_extractor(ans, drug_set)
            
            # 只在前3个样本打印详情
            if i <= 3:
                print(f"\n模型回答: {ans[:200]}...")
                print(f"\n检索到的文献 URL:")
                for idx, ref_url in enumerate(refs, 1):
                    print(f"  [{idx}] {ref_url}")
                print(f"\n提取的药物: {pred_meds}")
                print(f"参考药物: {true_meds}")
                print(f"参考回答: {ex.get('reference_response', '')[:200]}...")
            
        except Exception as e:
            print(f"Error processing example {i}:", e)
            continue
            
        # 收集评估数据
        preds_texts.append(ans)
        refs_texts.append(ex.get("reference_response", ""))
        preds_drugs.append(pred_meds)
        refs_drugs.append(true_meds)

    print("\nCalculating metrics...")
    prec = precision(preds_drugs, refs_drugs)
    rec = recall(preds_drugs, refs_drugs)
    f1 = f1_score(preds_drugs, refs_drugs)
    bert_res = compute_bertscore(preds_texts, refs_texts, lang="en")

    results = {
        "dataset": args.dataset,
        "n_examples": len(preds_texts),
        "drug_metrics": {
            "precision": prec,
            "recall": rec,
            "f1": f1
        },
        "bertscore": bert_res
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done. Results:\n", json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
