# -*- coding: utf-8 -*-
"""
answer_eva.py
RAG -> 生成 -> 严格法官 + 无标注客观指标（NLI支撑率/矛盾率、引用有效率、新近性）+ 统一量化分(RAG-QS)

运行示例：
  export DEEPSEEK_API_KEY=sk-xxx
  export JUDGE_MODEL=deepseek-chat
  export GEN_MODEL=deepseek-chat
  python answer_eva.py --queries "First-line therapy for CAP in adults?" "Contraindications of isotretinoin?"
  printf "Q1\nQ2\n" > queries.txt
  python answer_eva.py --queries_file queries.txt --out judge_report.jsonl --repeats 2
"""

from __future__ import annotations
import os, sys, re, json, argparse
from typing import List, Dict, Any, Tuple, Optional
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- 你的检索 ----
from retriever import retrieve  # 确保与项目内路径一致

# ---- 你的生成（若无则用兜底实现）----
try:
    from your_module_with_answer import answer  # 如果你已有正式实现，替换为真实模块
except Exception:
    from openai import OpenAI
    def answer(query: str) -> Tuple[str, List[str]]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        cli = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com"))
        ctx = retrieve(query) or []
        ctx_lines, refs = [], []
        for i, c in enumerate(ctx, 1):
            try:
                t = c[0][0]; u = c[0][1].get("url","")
                if t: ctx_lines.append(f"[{i}] {t}")
                refs.append(u)
            except Exception:
                pass
        context_block = "\n\n".join(ctx_lines) if ctx_lines else "[1] No evidence retrieved."
        prompt = f"""You are a medical assistant. Answer based only on context:

{context_block}

Question: {query}

If the question is not in English, take the translation of context into account.
Provide references as [1], [2] matching context.
"""
        r = cli.chat.completions.create(
            model=os.getenv("GEN_MODEL","deepseek-chat"),
            messages=[{"role":"user","content":prompt}],
            temperature=0
        ).choices[0].message.content
        return (r or "").strip(), refs

# ---- 法官（严格版）----
from judge import LLMJudge

# ---- NLI 支撑/矛盾率（独立于法官，客观度量）----
try:
    from transformers import pipeline
    _NLI = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)
except Exception:
    _NLI = None

def _std_ctx(raw_ctx: List[Any]) -> List[Dict[str, Any]]:
    from judge import LLMJudge as _J
    return _J.standardize_context(raw_ctx)

def _split_sents(text: str) -> List[str]:
    return [s for s in re.split(r'(?<=[.!?。！？])\s+', text.strip()) if s.strip()]

def _normalize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fa5\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def _best_ctx(sent: str, ctx_texts: List[str]) -> str:
    st = set(_normalize(sent))
    def score(t: str): return len(st & set(_normalize(t)))
    return max(ctx_texts, key=score) if ctx_texts else ""

def nli_support_contra(answer_text: str, ctx_texts: List[str]) -> Dict[str, float]:
    # 无 NLI 或无答案：直接 0
    if not _NLI or not answer_text:
        return {"SupportRate": 0.0, "ContradictionRate": 0.0}

    sents = _split_sents(answer_text)
    if not sents:
        return {"SupportRate": 0.0, "ContradictionRate": 0.0}

    sup = contra = total = 0
    for s in sents:
        ctx = _best_ctx(s, ctx_texts)
        if not ctx:
            continue
        # 适度裁剪，避免超长
        prem = ctx[:800]
        hypo = s[:300]
        try:
            # ★ 关键：用“批量输入(list)”强制返回 list
            out = _NLI([{"text": prem, "text_pair": hypo}], truncation=True)
            # 兼容返回结构：list[dict] 或 dict
            item = out[0] if isinstance(out, list) else out
            label = item.get("label", "")
        except Exception:
            # 任何异常都跳过这一句
            continue

        if not label:
            continue
        total += 1
        if label.upper() == "ENTAILMENT":
            sup += 1
        elif label.upper() == "CONTRADICTION":
            contra += 1
        # NEUTRAL 不计分

    if total == 0:
        return {"SupportRate": 0.0, "ContradictionRate": 0.0}
    return {"SupportRate": sup / total, "ContradictionRate": contra / total}


def citation_stats(ans_text: str, refs_urls: List[str], ctx_len: int) -> Dict[str, float]:
    # 解析答案里的 [n] 引用；检测是否越界
    idxs = [int(x) for x in re.findall(r"\[(\d+)\]", ans_text)]
    in_range = [i for i in idxs if 1 <= i <= max(1, ctx_len)]
    out_range = len(idxs) - len(in_range)
    resolvable = sum(1 for u in refs_urls if u and (u.startswith("http") or u.startswith("pmid:")))
    return {
        "CitationCount": len(idxs),
        "CitationOutOfRange": float(out_range),
        "CitationResolvable%": (resolvable / max(1,len(refs_urls))) if refs_urls else 0.0
    }

def freshness_estimate(urls: List[str], current_year: int = 2025) -> Dict[str, float]:
    years = []
    for u in urls:
        m = re.search(r"(19|20)\d{2}", u or "")
        if m: years.append(int(m.group(0)))
    if not years: return {"Freshness%_last5y": 0.0}
    fresh = sum(1 for y in years if current_year - y <= 5)
    return {"Freshness%_last5y": fresh/len(years)}

# ---- 单一量化指标：RAG-QS (0-100) ----
def _rag_qs_from_details(details: List[Dict[str, Any]]) -> float:
    def clamp01(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except:
            return 0.0

    scores = []
    for d in details:
        F_j = 20 * float(d.get("faithfulness", 0))  # 0-100
        R_j = 20 * float(d.get("relevance", 0))
        S_j = 20 * float(d.get("safety", 0))
        sup = clamp01(d.get("SupportRate", 0))
        con = clamp01(d.get("ContradictionRate", 0))
        NLI_j = 100 * max(0.0, sup - con)

        cite_res = clamp01(d.get("CitationResolvable%", 0))
        cc = float(d.get("CitationCount", 0) or 0)
        co = float(d.get("CitationOutOfRange", 0) or 0)
        CiteRes_j = 100 * cite_res
        CiteInt_j = 100 * (1.0 - min(1.0, (co / max(1.0, cc)))) if cc > 0 else 100.0  # 没引用不扣

        fresh = clamp01(d.get("Freshness%_last5y", 0))
        Fresh_j = 100 * fresh

        rag_qs = (0.30*F_j + 0.20*NLI_j + 0.15*R_j + 0.15*S_j
                  + 0.05*CiteRes_j + 0.05*CiteInt_j + 0.10*Fresh_j)
        scores.append(rag_qs)

    return sum(scores)/len(scores) if scores else 0.0

def evaluate_queries(queries: List[str], repeats: int=2, out_path: Optional[str]=None) -> Dict[str, Any]:
    judge = LLMJudge(model=os.getenv("JUDGE_MODEL","deepseek-chat"), temperature=0.0)
    details: List[Dict[str, Any]] = []

    for q in queries:
        raw_ctx = retrieve(q) or []
        ctx_items = _std_ctx(raw_ctx)
        ctx_texts = [c["text"] for c in ctx_items]
        ans_text, ans_refs = answer(q)

        # 严格法官（多次取均值）
        j = judge.judge_average(q, ans_text, ctx_items, repeats=repeats)

        # NLI 支撑/矛盾
        nli = nli_support_contra(ans_text, ctx_texts)

        # 引用质量 & 新近性
        cites = citation_stats(ans_text, ans_refs, ctx_len=len(ctx_items))
        fresh = freshness_estimate(ans_refs)

        row = {
            "query": q,
            "answer": ans_text,
            "refs": ans_refs,
            "faithfulness": j["faithfulness"],
            "relevance": j["relevance"],
            "safety": j["safety"],
            "overall": j["overall"],
            "claims": j.get("claims", [])[:6],
            "justifications": j.get("justifications", []),
            "SupportRate": nli["SupportRate"],
            "ContradictionRate": nli["ContradictionRate"],
            **cites,
            **fresh
        }
        details.append(row)

        print(f"[OK] {q}\n"
              f"  -> overall={row['overall']:.2f} (faith={row['faithfulness']:.2f}, rel={row['relevance']:.2f}, safe={row['safety']:.2f}) | "
              f"NLI(sup={row['SupportRate']:.2f}, contra={row['ContradictionRate']:.2f}) | "
              f"cit(res={row['CitationResolvable%']:.2f}, outRange={int(row['CitationOutOfRange'])})")

    # 汇总
    def mean(field: str) -> float:
        vals = [d[field] for d in details if isinstance(d[field], (int,float))]
        return sum(vals)/len(vals) if vals else 0.0

    summary = {
        "N": len(details),
        "Faithfulness_mean": mean("faithfulness"),
        "Relevance_mean": mean("relevance"),
        "Safety_mean": mean("safety"),
        "Overall_mean": mean("overall"),
        "NLI_SupportRate_mean": mean("SupportRate"),
        "NLI_ContradictionRate_mean": mean("ContradictionRate"),
        "CitationResolvable%_mean": mean("CitationResolvable%"),
        "CitationOutOfRange_mean": mean("CitationOutOfRange"),
        "Freshness%_last5y_mean": mean("Freshness%_last5y")
    }

    # 统一量化分
    summary["RAG_QS_mean"] = _rag_qs_from_details(details)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(out_path + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"summary": summary, "details": details}

def _load_queries_from_file(p: str) -> List[str]:
    qs: List[str] = []
    if p.endswith(".txt"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s: qs.append(s)
    elif p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, str): qs.append(obj)
                    elif isinstance(obj, dict) and "query" in obj: qs.append(str(obj["query"]))
                except Exception:
                    pass
    else:
        raise ValueError("仅支持 .txt 或 .jsonl")
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="*", default=None)
    ap.add_argument("--queries_file", type=str, default=None)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if not args.queries and not args.queries_file:
        print("请通过 --queries 或 --queries_file 提供问题列表。"); sys.exit(1)
    queries = _load_queries_from_file(args.queries_file) if args.queries_file else args.queries
    evaluate_queries(queries=queries, repeats=args.repeats, out_path=args.out)

if __name__ == "__main__":
    main()


# 1) 配置环境
# export DEEPSEEK_API_KEY=sk-ea1c109801b24fd5aa96b3d82d4193cb
# # 可选：设置不同模型
# export JUDGE_MODEL=deepseek-chat
# export GEN_MODEL=deepseek-chat
#
# # 2) 直接传问题
# python answer_eva.py --queries "First-line therapy for CAP in adults?" "Contraindications of isotretinoin?"
#
# # 3) 从文件读取
# printf "First-line therapy for CAP in adults?\nContraindications of isotretinoin?\n" > queries.txt
# python answer_eva.py --queries_file queries.txt --out judge_report.jsonl --repeats 2

