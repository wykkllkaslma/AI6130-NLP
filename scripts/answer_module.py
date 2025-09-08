from openai import OpenAI
from scripts.retriever import retrieve

client = OpenAI(api_key='sk-92727ffa636142f183bf63bb42f09bd6', base_url='https://api.deepseek.com')

def answer(query):
    ctx = retrieve(query)
    context_text = "\n\n".join([f"[{i+1}] {c[0][0]}" for i, c in enumerate(ctx)])
    refs = [c[0][1]["url"] for c in ctx]
    prompt = f"""You are a medical assistant. Answer based only on context:

{context_text}

Question: {query}

If the question is not in English. Your should take the translation of context into account.
And your answer should be based on the language of Question. 

Provide references as [1], [2] matching context.
"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, refs