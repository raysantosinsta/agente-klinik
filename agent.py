import os
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from contextvars import ContextVar
import urllib.request
import urllib.error

# Variável de contexto para identificar o tenant na execução da Tool
current_tenant: ContextVar[str] = ContextVar('current_tenant', default='')

# Define o estado do agente
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tenant_id: str
    phone_number: str
    intention: str
    human_intervention_needed: bool

# Configuração do LLM (Groq)
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.2)

# --- DEFINIÇÃO DAS FERRAMENTAS (TOOL CALLING) ---

@tool
def consultar_base_conhecimento(pergunta: str) -> str:
    """
    Consulta os documentos oficiais (PDFs, regras, preços, serviços e perguntas frequentes) da clínica/empresa.
    Use SEMPRE esta ferramenta antes de responder a qualquer dúvida sobre os serviços oferecidos, regras, modelos de atendimento, valores ou especialidades da empresa.
    """
    tenant = current_tenant.get()
    backend_url = os.getenv("BACKEND_URL", "http://localhost:3001")
    try:
        payload = json.dumps({"businessId": tenant, "query": pergunta}).encode("utf-8")
        req = urllib.request.Request(
            f"{backend_url}/chat/context",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
            business_name = data.get("businessName", "Empresa")
            context = data.get("context", "")
            services = data.get("services", "")

            result = f"=== Informações Oficiais da {business_name} ===\n"
            if context:
                result += f"Trechos do Documento Oficial (PDF):\n{context}\n\n"
            if services:
                result += f"Serviços Cadastrados:\n{services}\n"
            return result
    except Exception as e:
        return f"Erro ao acessar base de conhecimento: {str(e)}"

@tool
def consultar_disponibilidade_agenda(data: str) -> str:
    """
    Consulta a disponibilidade de agenda para uma data específica.
    Formato esperado da data: 'YYYY-MM-DD'.
    Retorna os horários disponíveis.
    """
    return f"Para a data {data}, temos disponibilidade nos seguintes horários: 09:00, 14:00 e 16:30."

@tool
def calcular_score_lead(nivel_interesse: int, servico_escolhido: str) -> str:
    """
    Calcula e atribui uma pontuação (score) técnica para o lead com base no nível de interesse (1 a 10) e no serviço escolhido.
    Use esta ferramenta quando o cliente demonstrar intenção clara de agendamento ou contratação.
    """
    score = nivel_interesse * 10
    if servico_escolhido:
        score += 15
    return f"Score atualizado para {score}. Lead qualificado."

@tool
def atualizar_etapa_crm(nova_etapa: str) -> str:
    """
    Atualiza a etapa do lead no CRM.
    Etapas válidas: 'Frio', 'Morno', 'Agendado'.
    """
    return f"Etapa do CRM atualizada com sucesso para: {nova_etapa}."

tools = [consultar_base_conhecimento, consultar_disponibilidade_agenda, calcular_score_lead, atualizar_etapa_crm]
llm_with_tools = llm.bind_tools(tools)

import logging

logger = logging.getLogger("ai_agent")

# --- NÓS DO GRAFO ---

async def analyze_intention(state: AgentState) -> AgentState:
    """Analisa se o lead quer agendar, tirar dúvida, ou se está irritado."""
    last_message = state["messages"][-1].content
    logger.info("  🔍 [Etapa 1: Análise de Intenção] Analisando mensagem do usuário...")

    sys_prompt = (
        "Analise a última mensagem do usuário.\n"
        "Se o usuário exigir explicitamente falar com um humano/atendente, responda 'HUMAN'.\n"
        "Caso contrário, responda 'NORMAL'."
    )

    response = await llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=last_message)])

    if "HUMAN" in response.content:
        logger.info("  ⚠️ [Etapa 1: Intenção] Cliente solicitou falar com HUMANO! Escalando atendimento...")
        return {"intention": "escalate", "human_intervention_needed": True}

    logger.info("  ➡️ [Etapa 1: Intenção] Mensagem normal identificada. Encaminhando para o LLM...")
    return {"intention": "general", "human_intervention_needed": False}

async def generate_response(state: AgentState) -> AgentState:
    """Nó para gerar a resposta, podendo decidir chamar uma ferramenta."""
    logger.info("  🧠 [Etapa 2: LLM Groq] Enviando histórico e gerando raciocínio...")

    sys_prompt = SystemMessage(content=(
        "Você é um assistente virtual profissional e inteligente da empresa.\n"
        "DIRETRIZES FUNDAMENTAIS:\n"
        "1. Se o cliente perguntar sobre serviços, soluções, preços, catálogo, dúvidas técnicas ou regras da empresa, você DEVE OBRIGATORIAMENTE chamar a ferramenta 'consultar_base_conhecimento'.\n"
        "2. Responda única e exclusivamente com base no conteúdo retornado pela ferramenta oficial. NUNCA invente serviços, regras ou procedimentos que não pertençam à empresa.\n"
        "3. Se o cliente quiser marcar uma reunião, consulta ou horário, use a ferramenta 'consultar_disponibilidade_agenda'.\n"
        "4. Mantenha um tom prestativo, profissional e objetivo ideal para WhatsApp."
    ))

    messages_to_send = [sys_prompt] + list(state["messages"])

    # O LLM decide se responde diretamente ou se chama uma ferramenta
    response = await llm_with_tools.ainvoke(messages_to_send)

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        logger.info(f"  🛠️ [Etapa 2: Decisão do LLM] Solicitou execução de {len(tool_calls)} ferramenta(s): {[t['name'] for t in tool_calls]}")
    else:
        logger.info(f"  💬 [Etapa 2: Decisão do LLM] Resposta direta gerada com sucesso.")

    return {"messages": [response]}

async def execute_tools(state: AgentState) -> AgentState:
    """Executa as ferramentas solicitadas pelo LLM e devolve o resultado."""
    last_message = state["messages"][-1]

    tool_messages = []
    # Itera sobre todas as tool_calls geradas pelo LLM
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info(f"  ⚙️ [Etapa 3: Executando Tool] {tool_name} com parâmetros: {tool_args}")

        # Encontra a ferramenta correspondente
        matched_tool = next((t for t in tools if t.name == tool_name), None)
        if matched_tool:
            # Executa a ferramenta
            try:
                result = matched_tool.invoke(tool_args)
                logger.info(f"  ✨ [Etapa 3: Resultado da Tool] {tool_name} -> {result}")
            except Exception as e:
                result = f"Erro ao executar a ferramenta: {str(e)}"
                logger.error(f"  ❌ [Etapa 3: Erro na Tool] {tool_name} -> {result}")
        else:
            result = f"Ferramenta {tool_name} não encontrada."
            logger.warning(f"  ⚠️ [Etapa 3: Tool Não Encontrada] {tool_name}")

        # Cria a mensagem de resposta da ferramenta
        tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        tool_messages.append(tool_msg)

    logger.info("  🔁 [Etapa 3: Retornando ao LLM] Devolvendo dados das ferramentas para o modelo sintetizar a resposta...")
    return {"messages": tool_messages}

# --- LÓGICA DE ROTEAMENTO ---

def route_intention(state: AgentState) -> str:
    if state.get("human_intervention_needed", False):
        return "escalate"
    return "respond"

def should_continue_or_tool(state: AgentState) -> str:
    """Decide se o fluxo termina ou se as ferramentas precisam ser executadas."""
    last_message = state["messages"][-1]

    # Se o LLM retornou tool_calls, vamos para o nó de execução de ferramentas
    if getattr(last_message, "tool_calls", None):
        return "tools"

    # Caso contrário, ele apenas respondeu e podemos finalizar
    return "end"

# --- CONSTRUÇÃO DO GRAFO ---
workflow = StateGraph(AgentState)

workflow.add_node("analyzer", analyze_intention)
workflow.add_node("responder", generate_response)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("analyzer")

# analyzer -> (respond | escalate)
workflow.add_conditional_edges("analyzer", route_intention, {"respond": "responder", "escalate": END})

# responder -> (tools | END)
workflow.add_conditional_edges("responder", should_continue_or_tool, {"tools": "tools", "end": END})

# tools -> responder (volta pro LLM avaliar a resposta da ferramenta e falar com o cliente)
workflow.add_edge("tools", "responder")

app_graph = workflow.compile()

async def process_message(tenant_id: str, phone_number: str, message: str) -> dict:
    current_tenant.set(tenant_id)
    logger.info(f"🚀 [LangGraph Iniciado] Tenant: {tenant_id} | De: {phone_number}")
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "tenant_id": tenant_id,
        "phone_number": phone_number,
        "intention": "",
        "human_intervention_needed": False
    }

    final_state = await app_graph.ainvoke(initial_state)

    reply = None
    if not final_state.get("human_intervention_needed") and len(final_state["messages"]) > 1:
        # A última mensagem será do LLM (após possivelmente ter passado pelas ferramentas)
        reply = final_state["messages"][-1].content

    logger.info(f"🏁 [LangGraph Finalizado] Resposta pronta ({len(reply) if reply else 0} caracteres)")
    return {
        "reply": reply,
        "human_intervention_needed": final_state.get("human_intervention_needed", False)
    }
