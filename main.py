import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import do LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------------
# Configuração da API Key
# ------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não está definida!")

# ------------------------------
# Prompt do sistema
# ------------------------------
SYSTEM_PROMPT = """
Você é um assistente de IA especializado em identificação macroscópica de minerais.
Use uma formatação para terminais linux.
Receba a descrição e retorne 2–3 minerais prováveis, explicando características diagnósticas e como diferenciá-los.
Se houver incerteza, sugira testes adicionais e ressalte limites da análise.
"""

# ------------------------------
# Inicialização do FastAPI
# ------------------------------
app = FastAPI()

# ------------------------------
# CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Servir frontend em /frontend
# ------------------------------
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# ------------------------------
# Modelo de request
# ------------------------------
class MineralRequest(BaseModel):
    descricao: str

# ------------------------------
# Criar chain do LangChain
# ------------------------------
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

# ------------------------------
# Endpoint POST /identificar
# ------------------------------
@app.post("/identificar")
def identificar_mineral(request: MineralRequest):
    descricao = request.descricao.strip()
    if len(descricao) < 5:
        return {"erro": "Descrição muito curta, descreva melhor o mineral."}
    try:
        resposta = chain.invoke({"query": descricao})
        return {"resposta": resposta}
    except Exception as e:
        return {"erro": f"Erro interno: {str(e)}"}

# ------------------------------
# Endpoint teste rápido
# ------------------------------
@app.get("/teste")
def teste():
    return {"status": "Backend funcionando!"}
