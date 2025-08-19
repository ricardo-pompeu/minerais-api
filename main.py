from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import json

API_KEY = "SUA_CHAVE_GOOGLE"

SYSTEM_PROMPT = """
Você é um assistente de IA especializado em identificação macroscópica de minerais.

Retorne **sempre em JSON**, no seguinte formato:

{
  "minerais": [
    {
      "nome": "Nome do mineral",
      "caracteristicas": ["item 1", "item 2", "item 3"],
      "diferenciacao": "Como diferenciar esse mineral"
    }
  ],
  "testes": ["teste 1", "teste 2"],
  "limites": ["limite 1", "limite 2"]
}

⚠️ Não use markdown, não use código, apenas JSON válido.
"""

app = FastAPI()

class MineralRequest(BaseModel):
    descricao: str

def criar_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=API_KEY
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    return prompt | llm | StrOutputParser()

chain = criar_chain()

def formatar_html(json_str: str) -> str:
    """Converte o JSON gerado pelo modelo em HTML bonito"""
    try:
        dados = json.loads(json_str)
    except:
        return f"<p class='erro'>Erro ao interpretar resposta da IA: {json_str}</p>"

    html = "<div class='analise'>"
    html += "<h1>🔎 Possibilidades de Identificação</h1>"

    # Minerais
    for idx, mineral in enumerate(dados.get("minerais", []), start=1):
        html += f"<h2>🟢 Mineral {idx} — {mineral['nome']}</h2>"
        html += "<ul>"
        for c in mineral.get("caracteristicas", []):
            html += f"<li>{c}</li>"
        html += "</ul>"
        html += f"<p><strong>Como diferenciar:</strong> {mineral.get('diferenciacao', '')}</p>"

    # Testes adicionais
    if "testes" in dados:
        html += "<h2>🧪 Testes adicionais recomendados</h2><ul>"
        for t in dados["testes"]:
            html += f"<li>{t}</li>"
        html += "</ul>"

    # Limites
    if "limites" in dados:
        html += "<h2>⚠️ Limites da análise</h2><ul>"
        for l in dados["limites"]:
            html += f"<li>{l}</li>"
        html += "</ul>"

    html += "</div>"
    return html

@app.post("/identificar")
def identificar_mineral(request: MineralRequest):
    if len(request.descricao.strip()) < 5:
        return {"erro": "Descrição muito curta, descreva melhor o mineral."}
    resposta_json = chain.invoke({"query": request.descricao})
    resposta_html = formatar_html(resposta_json)
    return {"resposta": resposta_html}
