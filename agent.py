import os
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

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
def consultar_disponibilidade_agenda(data: str) -> str:
    """
    Consulta a disponibilidade de agenda para uma data específica.
    Formato esperado da data: 'YYYY-MM-DD'.
    Retorna os horários disponíveis.
    """
    # TODO: Integrar com a base do NestJS/PostgreSQL
    # Simulação:
    return f"Para a data {data}, temos disponibilidade às 09:00, 14:00 e 16:30."

@tool
def calcular_score_lead(nivel_interesse: int, servico_escolhido: str) -> str:
    """
    Calcula e atribui uma pontuação (score) técnica para o lead com base no nível de interesse (1 a 10) e no serviço escolhido.
    Use esta ferramenta quando o cliente demonstrar intenção clara de agendamento ou contratação.
    """
    # TODO: Regra de negócio interna complexa
    score = nivel_interesse * 10
    if servico_escolhido.lower() in ['harmonização', 'botox']:
        score += 20
    return f"Score atualizado para {score}. Lead qualificado."

@tool
def atualizar_etapa_crm(nova_etapa: str) -> str:
    """
    Atualiza a etapa do lead no CRM.
    Etapas válidas: 'Frio', 'Morno', 'Agendado'.
    """
    # TODO: Disparar atualização no banco relacional (PostgreSQL)
    return f"Etapa do CRM atualizada com sucesso para: {nova_etapa}."

tools = [consultar_disponibilidade_agenda, calcular_score_lead, atualizar_etapa_crm]
llm_with_tools = llm.bind_tools(tools)

# --- NÓS DO GRAFO ---

async def analyze_intention(state: AgentState) -> AgentState:
    """Analisa se o lead quer agendar, tirar dúvida, ou se está irritado."""
    last_message = state["messages"][-1].content

    sys_prompt = (
        "Analise a última mensagem do usuário.\n"
        "Se o usuário exigir explicitamente falar com um humano/atendente, responda 'HUMAN'.\n"
        "Caso contrário, responda 'NORMAL'."
    )

    response = await llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=last_message)])

    if "HUMAN" in response.content:
        return {"intention": "escalate", "human_intervention_needed": True}

    return {"intention": "general", "human_intervention_needed": False}

async def generate_response(state: AgentState) -> AgentState:
    """Nó para gerar a resposta, podendo decidir chamar uma ferramenta."""

    sys_prompt = SystemMessage(content=(
        "Você é um assistente virtual B2B de qualificação de leads e atendimento.\n"
        "Sua função é tirar dúvidas, agendar horários e qualificar o cliente.\n"
        "Se precisar de horários, calcular o score do lead, ou mudar o estágio no CRM, use as ferramentas disponíveis."
    ))

    messages_to_send = [sys_prompt] + list(state["messages"])

    # O LLM decide se responde diretamente ou se chama uma ferramenta
    response = await llm_with_tools.ainvoke(messages_to_send)
    return {"messages": [response]}

async def execute_tools(state: AgentState) -> AgentState:
    """Executa as ferramentas solicitadas pelo LLM e devolve o resultado."""
    last_message = state["messages"][-1]

    tool_messages = []
    # Itera sobre todas as tool_calls geradas pelo LLM
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Encontra a ferramenta correspondente
        matched_tool = next((t for t in tools if t.name == tool_name), None)
        if matched_tool:
            # Executa a ferramenta
            try:
                result = matched_tool.invoke(tool_args)
            except Exception as e:
                result = f"Erro ao executar a ferramenta: {str(e)}"
        else:
            result = f"Ferramenta {tool_name} não encontrada."

        # Cria a mensagem de resposta da ferramenta
        tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        tool_messages.append(tool_msg)

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

    return {
        "reply": reply,
        "human_intervention_needed": final_state.get("human_intervention_needed", False)
    }
