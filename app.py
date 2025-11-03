from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import os

# ========= 1. 啟動 Flask =========
app = Flask(__name__)

# ========= 2. 載入模型與資料庫 =========
class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

embedding_model = CustomE5Embedding(model_name="intfloat/multilingual-e5-small")
db = FAISS.load_local("faiss_db", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# ========= 3. LLM設定 =========
# 記得在 Render 設定環境變數 OPENAI_API_KEY
api_key = os.getenv("GROQ_API_KEY")
base_url = "https://api.groq.com/openai/v1"
client = OpenAI(api_key=api_key, base_url=base_url)
model = "llama-3.1-8b-instant"

# ========= 4. Prompt =========
system_prompt = """
你是一位來自台灣且講中文的專業、親切的金融理財顧問，請根據資料來回應使用者的問題。
1.請親切、簡潔並具體建議。
2.使用檢索到的資料和之前的對話回答。
3.請用台灣習慣的繁體中文回應，如果回答出現英文的都要翻譯成中文。
4.不夾帶表情符號。
5.資料不足請老實告知。
6.不清楚的問題請請使用者再描述一次。
7.語氣友善、簡潔。
8.打招呼時不用檢索。
9.不要畫線。
10.可以使用表格輔助。
"""

prompt_template = """
你是台灣人，是一位專業的「金融理財顧問」，使用台灣常用的中文回覆問題，
請根據這些資訊清楚且易懂地回答使用者的問題：

{retrieved_chunks}

使用者的問題是：
{question}

請用台灣習慣的中文回答，提供實用的建議或說明。
"""

# ========= 5. RAG 主邏輯 =========
def chat_with_rag(user_input):
    docs = retriever.get_relevant_documents(user_input)
    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt_template.format(retrieved_chunks=retrieved_chunks, question=user_input)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]
    )
    answer = response.choices[0].message.content
    return answer

# ========= 6. API 路由 =========
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("question", "")
    if not user_input:
        return jsonify({"error": "缺少問題欄位"}), 400
    try:
        answer = chat_with_rag(user_input)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "RAG Financial Assistant API is running ✅"

# ========= 7. 啟動 =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
