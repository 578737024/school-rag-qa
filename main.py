import re
import numpy as np
import pandas as pd
import subprocess
from sentence_transformers import SentenceTransformer

# =========================
# 1. 配置 Ollama
# =========================
OLLAMA_PATH = "你的ollama.exe绝对路径"
MODEL_NAME = "你的本地模型 如：deepseek-r1:8b"

# =========================
# 2. 加载 RAG embedding 模型
# =========================
embed_model = SentenceTransformer("BAAI/bge-small-zh")

# =========================
# 3. 加载规则文本
# =========================
def load_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)", text, re.S)
    docs, ids = [], []
    for num, content in matches:
        content = content.strip().replace("\n", "")
        if content:
            docs.append(content)
            ids.append(num)
    return docs, ids

docs, ids = load_rules("data/school.txt")
print(f"✅ 共加载 {len(docs)} 条规则")

# =========================
# 4. 生成 embedding
# =========================
embeddings = embed_model.encode(docs, normalize_embeddings=True)

# =========================
# 5. RAG 检索函数（带上下文长度优化）
# =========================
MAX_CONTEXT_CHARS = 1000  # 上下文总字符数上限
TOP_K = 10                # 先取 top_k，再截断到字符长度

def retrieve(query, top_k=3):
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    scores = np.dot(embeddings, q_vec.T).squeeze()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    selected_docs = []
    selected_ids = []
    char_count = 0
    for i in top_indices:
        doc = docs[i]
        if char_count + len(doc) > MAX_CONTEXT_CHARS:
            break
        selected_docs.append(doc)
        selected_ids.append(ids[i])
        char_count += len(doc)
    return selected_docs, selected_ids  # 返回内容和编号

# =========================
# 6. 调用 Ollama 做最终判断
# =========================
def ollama_judge(query, context_rules):
    prompt = f"""你是一个学校规则问答助手，请根据以下校规严格回答问题：

【校规】：
{context_rules}

【问题】：
{query}

请严格只输出两行：
1. 允许 或 不允许
2. 简短理由（直接引用校规）
不要输出任何多余文字或思考过程。
"""
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", MODEL_NAME],
            input=prompt.encode(),
            capture_output=True,
            timeout=60
        )
        output = result.stdout.decode().strip()
        if not output:
            return "不确定", "模型未返回结果"
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        # 取最后两行作为判定和原因
        decision = lines[-2] if len(lines) >= 2 else "不确定"
        reason = lines[-1] if len(lines) >= 1 else ""
        return decision, reason
    except Exception as e:
        return "❌ Ollama 出错", str(e)

# =========================
# 7. 缓存机制
# =========================
cache = {}  # key: 问题小写字符串，value: (decision, reason, source)

def ask(query):
    key = query.strip().lower()
    if key in cache:
        return cache[key]

    top_rules, top_ids = retrieve(query, top_k=TOP_K)
    context = "\n".join(top_rules)
    decision, reason = ollama_judge(query, context)
    source = f"第{top_ids[0]}条校规" if top_ids else "无"
    cache[key] = (decision, reason, source)
    return decision, reason, source

# =========================
# 8. 主程序
# =========================
print("\n📚 AI校规问答系统（混合模式 RAG + Ollama deepseek-r1:8B，缓存+上下文截断优化 + 来源输出）")

while True:
    q = input("\n请输入问题（q退出）：")
    if q.lower() == "q":
        break
    decision, reason, source = ask(q)
    print("\n🤖 回答：")
    print(f"判定：{decision}")
    print(f"原因：{reason}")
    print(f"来源：{source}")
