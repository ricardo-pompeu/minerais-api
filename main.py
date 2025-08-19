import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Lendo a chave da variável de ambiente
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não está definida!")

SYSTEM_PROMPT = """
Você é um assistente de IA especializado em identificação macroscópica de minerais.
Use uma formatação para terminais linux.
Receba a descrição e retorne 2–3 minerais prováveis, explicando características diagnósticas e como diferenciá-los.
Se houver incerteza, sugira testes adicionais e ressalte limites da análise.
"""

app = FastAPI()

# Modelo para requisição
class MineralRequest(BaseModel):
    descricao: str

# Criando chain
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

@app.post("/identificar")
def identificar_mineral(request: MineralRequest):
    descricao = request.descricao.strip()
    if len(descricao) < 5:
        return {"erro": "Descrição muito curta, descreva melhor o mineral."}
    resposta = chain.invoke({"query": descricao})
    return {"resposta": resposta}
