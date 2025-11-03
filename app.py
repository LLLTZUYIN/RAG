from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ========= 1. 啟動 Flask =========
app = Flask(__name__)

# ========= 2. 載入向量資料庫 =========
class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

embedding_model = CustomE5Embedding(model_name="intfloat/multilingual-e5-small")

# 載入本地向量資料庫（記得要有 "faiss_db" 資料夾）
db = FAISS.load_local("faiss_db", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# ========= 3. RAG 檢索邏輯（不包含 LLM） =========
def retrieve_context(user_input):
    docs = retriever.get_relevant_documents(user_input)
    retrieved_chunks = [doc.page_content for doc in docs]
    return retrieved_chunks

# ========= 4. API 路由 =========
@app.route("/ask", methods=["POST"])
def ask():
    """
    使用方式：
    POST /ask
    {
        "question": "請問ETF是什麼？"
    }
    回傳：
    {
        "context": [...相關文檔...],
        "question": "請問ETF是什麼？"
    }
    """
    data = request.get_json()
    user_input = data.get("question", "")

    if not user_input:
        return jsonify({"error": "缺少問題欄位"}), 400

    try:
        retrieved_chunks = retrieve_context(user_input)
        return jsonify({
            "question": user_input,
            "context": retrieved_chunks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "RAG Context Retrieval API is running ✅"

# ========= 5. 啟動 =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
