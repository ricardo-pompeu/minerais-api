import os
from flask import Flask, request, jsonify, send_from_directory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Leia a chave do ambiente (configure no Render)
API_KEY = os.getenv("GOOGLE_API_KEY")

SYSTEM_PROMPT = """
Você é um assistente de IA especialista em identificação macroscópica de minerais.
Sua tarefa é receber uma descrição do mineral fornecida pelo usuário e sugerir 2 a 3 minerais mais prováveis com as respectivas fórmulas químicas ao lado do nome.
Para cada mineral sugerido, forneça:

1. Nome do mineral.
2. Principais características diagnósticas visíveis (cor, brilho, traço, clivagem, hábito, densidade, fratura, etc.).
3. Uma breve explicação de por que esse mineral é compatível com a descrição.

Responda de forma clara, objetiva e organizada, usando listas numeradas ou bullets quando necessário.
Não inclua minerais muito raros ou irrelevantes.
Sempre baseie suas sugestões em observações macroscópicas facilmente identificáveis.
"""

# Flask servirá também o frontend (mesma origem, sem CORS)
app = Flask(__name__, static_folder="frontend", static_url_path="")

# Inicializa LLM uma vez (startup)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{query}")
])

@app.get("/")
def serve_index():
    # Servir o frontend
    return send_from_directory("frontend", "index.html")

@app.post("/identificar")
def identificar():
    data = request.get_json(silent=True) or {}
    descricao = (data.get("descricao") or "").strip()

    if not descricao or len(descricao) < 5:
        return jsonify(error="Descrição muito curta. Dê mais detalhes."), 400

    try:
        resposta_objeto = (prompt | llm).invoke({"query": descricao})
        resposta_texto = getattr(resposta_objeto, "content", str(resposta_objeto))
        return jsonify(resposta=resposta_texto)
    except Exception as e:
        # Log simples (Render mostra nos logs)
        print("Erro na geração:", repr(e))
        return jsonify(error="Falha ao processar a requisição."), 500

@app.get("/healthz")
def health():
    ok = bool(API_KEY)
    return jsonify(status="ok" if ok else "missing_api_key", has_api_key=ok)

# Para rodar localmente (opcional)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)
