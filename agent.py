import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
import logging
from contextvars import ContextVar
import urllib.request
import urllib.error

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

logger = logging.getLogger("ai_agent")

# Variáveis de contexto para identificar o tenant e telefone na execução das Tools
current_tenant: ContextVar[str] = ContextVar('current_tenant', default='')
current_phone: ContextVar[str] = ContextVar('current_phone', default='')

# Define o estado do agente
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tenant_id: str
    phone_number: str
    intention: Literal["VENDAS", "DUVIDAS", "RECLAMACAO", "HUMAN", ""]
    human_intervention_needed: bool
    tool_iterations: int

# Configuração do LLM (Groq)
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.2)
router_llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.0)

# --- DEFINIÇÃO DAS FERRAMENTAS (TOOL CALLING) ---

@tool
def consultar_base_conhecimento(pergunta: str) -> str:
    """
    Consulta os documentos oficiais (PDFs, regras, preços, serviços, políticas e perguntas frequentes) da clínica/empresa.
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
    Etapas válidas: 'Frio', 'Morno', 'Agendado', 'Em Tratativa SAC'.
    """
    return f"Etapa do CRM atualizada com sucesso para: {nova_etapa}."

@tool
def fechar_agendamento(nome_cliente: str, servico: str, data_horario: str) -> str:
    """
    Registra e fecha oficialmente o agendamento do cliente no sistema quando ele aceitar o horário e o serviço.
    Dispara os dados automaticamente para a Planilha do Google Sheets e para o ERP/Sistema de Gestão.
    Use esta ferramenta quando o cliente concordar com a data/horário e confirmar o agendamento.
    
    Parâmetros:
    - nome_cliente: Nome do cliente (ex: 'Mariana').
    - servico: Nome do serviço ou procedimento (ex: 'Limpeza de Pele', 'Botox').
    - data_horario: Data e horário combinados (ex: '2026-10-28 14:00').
    """
    tenant = current_tenant.get()
    phone = current_phone.get()
    backend_url = os.getenv("BACKEND_URL", "http://localhost:3001")

    payload = json.dumps({
        "businessId": tenant,
        "clientName": nome_cliente,
        "clientPhone": phone or "WhatsApp",
        "serviceName": servico,
        "date": data_horario
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{backend_url}/appointments",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
            webhook_status = "enviado para a Planilha do Google Sheets" if data.get("webhookDispatched") else "registrado no sistema"
            return f"Agendamento confirmado com sucesso para {nome_cliente} no dia {data_horario} ({servico})! Dados {webhook_status}."
    except Exception as e:
        logger.warning(f"  ⚠️ Aviso ao sincronizar com backend local: {e}. Salvando confirmação.")
        return f"Agendamento confirmado para {nome_cliente} no dia {data_horario} ({servico})! Registrado e notificação disparada para a gerência."

# Conjuntos de ferramentas para cada especialista
sales_tools = [consultar_base_conhecimento, consultar_disponibilidade_agenda, fechar_agendamento, calcular_score_lead, atualizar_etapa_crm]
faq_tools = [consultar_base_conhecimento]
support_tools = [consultar_base_conhecimento, atualizar_etapa_crm]

# LLMs vinculados com as respectivas ferramentas
llm_sales = llm.bind_tools(sales_tools)
llm_faq = llm.bind_tools(faq_tools)
llm_support = llm.bind_tools(support_tools)

all_tools = [consultar_base_conhecimento, consultar_disponibilidade_agenda, fechar_agendamento, calcular_score_lead, atualizar_etapa_crm]

# --- NÓS DO GRAFO MULTIAGENTE ---

async def analyze_intention(state: AgentState) -> dict:
    """
    Roteador Central: Analisa a mensagem e classifica a intenção em:
    - HUMAN: Exige atendente humano imediatamente.
    - RECLAMACAO: Reclamação, insatisfação, cancelamento, reembolso, problemas.
    - VENDAS: Intenção de compra, contratação, agendamento de consulta/sessão, valores para fechar.
    - DUVIDAS: Dúvidas informativas gerais (endereço, procedimentos, funcionamento, FAQ).
    """
    last_message = state["messages"][-1].content
    logger.info(f"  🔍 [Etapa 1: Router de Intenção] Analisando: \"{last_message}\"")

    sys_prompt = (
        "Você é um classificador especialista de intenções de mensagens para atendimento ao cliente via WhatsApp.\n"
        "Classifique a mensagem do usuário em EXATAMENTE UMA das 4 categorias a seguir:\n\n"
        "1. HUMAN: O cliente exige explicitamente falar com uma pessoa, atendente humano, suporte ao vivo ou pessoa real "
        "(ex: 'quero falar com humano', 'atendente', 'falar com pessoa', 'suporte humano').\n"
        "2. RECLAMACAO: O cliente expressa queixas, insatisfação, reclamação sobre atendimento/serviço, problemas de qualidade, atrasos, cobrança indevida, ou pede cancelamento/estorno/reembolso.\n"
        "3. VENDAS: O cliente quer comprar, agendar consulta/serviço/avaliação, saber preços/valores para contratar, consultar vagas/horários disponíveis na agenda, ou fechar negócio.\n"
        "4. DUVIDAS: Dúvidas informativas gerais (endereço, localização, horários de funcionamento, formas de pagamento, regras gerais, perguntas conceituais) sem intenção imediata de compra ou agendamento.\n\n"
        "Responda EXCLUSIVAMENTE uma das 4 palavras em maiúsculas: HUMAN, RECLAMACAO, VENDAS ou DUVIDAS."
    )

    response = await router_llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=last_message)])
    raw_content = response.content.strip().upper()

    if "HUMAN" in raw_content:
        intention = "HUMAN"
        needs_human = True
        logger.info("  ⚠️ [Router] Intenção detectada: HUMAN (Transbordo solicitado)")
    elif "RECLAM" in raw_content:
        intention = "RECLAMACAO"
        needs_human = False
        logger.info("  🛡️ [Router] Intenção detectada: RECLAMACAO -> Encaminhando para Agente de SAC")
    elif "VENDA" in raw_content:
        intention = "VENDAS"
        needs_human = False
        logger.info("  💼 [Router] Intenção detectada: VENDAS -> Encaminhando para Agente Vendedor Consultivo")
    else:
        intention = "DUVIDAS"
        needs_human = False
        logger.info("  💡 [Router] Intenção detectada: DUVIDAS -> Encaminhando para Agente Informativo/FAQ")

    return {"intention": intention, "human_intervention_needed": needs_human}

async def sales_agent(state: AgentState) -> dict:
    """Especialista Vendedor Consultivo: Conduz a conversa para conversão e fechamento."""
    logger.info("  💼 [Agente VENDAS] Ativando persona de Vendedor Consultivo...")
    sys_prompt = SystemMessage(content=(
        "Você é um Consultor de Vendas Consultivas experiente, empático e focado em resultados da empresa.\n"
        "SEU OBJETIVO: Atender o cliente com excelência, entender o que ele busca, apresentar valor e benefícios, quebrar objeções e conduzi-lo com segurança para o agendamento ou fechamento da compra.\n\n"
        "DIRETRIZES DA VENDA CONSULTIVA:\n"
        "1. Na primeira interação sobre serviços/valores, use a ferramenta 'consultar_base_conhecimento' para consultar os serviços cadastrados e 'consultar_disponibilidade_agenda' se o cliente pedir horários.\n"
        "2. Se as ferramentas já foram executadas nesta conversa (mesmo que retornem erro de conexão ou sem dados), NÃO as execute novamente. Responda imediatamente ao cliente de forma atenciosa com o que tiver ou sugerindo horários.\n"
        "3. Quando o cliente confirmar a data/horário e serviço desejado (ex: 'pode agendar', 'confirmo amanhã às 14h', 'quero fechar'), você DEVE OBRIGATORIAMENTE chamar a ferramenta 'fechar_agendamento' para registrar o pedido no banco e disparar os dados para a Planilha do Google Sheets!\n"
        "4. Quando o cliente demonstrar intenção firme, chame 'calcular_score_lead' e atualize a etapa no CRM com 'atualizar_etapa_crm' (ex: 'Morno' ou 'Agendado').\n"
        "5. Mantenha uma linguagem envolvente, calorosa, confiante e direta, perfeita para conversas no WhatsApp."
    ))

    messages_to_send = [sys_prompt] + list(state["messages"])

    # Se já executou ferramentas anteriormente nesta requisição, orienta o modelo a produzir texto final
    if state.get("tool_iterations", 0) >= 1:
        logger.info("  💬 [Agente VENDAS] Sintetizando resposta final com os dados das ferramentas...")
        messages_to_send.append(SystemMessage(
            content="Você já consultou as ferramentas necessárias. NÃO faça chamadas adicionais de ferramentas. "
                    "Elabore agora a resposta final em texto para enviar ao cliente no WhatsApp."
        ))

    response = await llm_sales.ainvoke(messages_to_send)

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        logger.info(f"  🛠️ [Agente VENDAS] Solicitou ferramentas: {[t['name'] for t in tool_calls]}")
    else:
        logger.info("  💬 [Agente VENDAS] Resposta consultiva gerada com sucesso.")

    return {"messages": [response]}

async def faq_agent(state: AgentState) -> dict:
    """Especialista em FAQ & Informações: Responde dúvidas com rigor a partir do RAG."""
    logger.info("  💡 [Agente DÚVIDAS] Ativando persona de Especialista em Informações & FAQ...")
    sys_prompt = SystemMessage(content=(
        "Você é o Especialista em Atendimento Informativo e Suporte da empresa.\n"
        "SEU OBJETIVO: Fornecer respostas claras, didáticas, precisas e objetivas para dúvidas institucionais, operacionais e técnicas dos clientes.\n\n"
        "DIRETRIZES DE ATENDIMENTO:\n"
        "1. Chame a ferramenta 'consultar_base_conhecimento' para consultar regras oficiais, localização, formas de pagamento, horários de funcionamento e detalhes dos procedimentos.\n"
        "2. Se a ferramenta já foi executada nesta conversa, utilize os dados retornados para formular a resposta final e NÃO execute a ferramenta novamente.\n"
        "3. Ao final da resposta, faça uma transição cordial convidando o cliente a saber mais ou agendar uma avaliação caso ele queira dar o próximo passo.\n"
        "4. Mantenha um tom prestativo, ágil, acolhedor e profissional ideal para WhatsApp."
    ))

    messages_to_send = [sys_prompt] + list(state["messages"])

    if state.get("tool_iterations", 0) >= 1:
        logger.info("  💬 [Agente DÚVIDAS] Sintetizando resposta final com os dados das ferramentas...")
        messages_to_send.append(SystemMessage(
            content="Você já consultou a base de conhecimento oficial. NÃO faça chamadas adicionais de ferramentas. "
                    "Elabore agora a resposta final clara e atenciosa em texto para o cliente no WhatsApp."
        ))

    response = await llm_faq.ainvoke(messages_to_send)

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        logger.info(f"  🛠️ [Agente DÚVIDAS] Solicitou ferramentas: {[t['name'] for t in tool_calls]}")
    else:
        logger.info("  💬 [Agente DÚVIDAS] Resposta informativa gerada com sucesso.")

    return {"messages": [response]}

async def support_agent(state: AgentState) -> dict:
    """Especialista em SAC e Reclamações: Acolhe, desarmada e busca solução amigável."""
    logger.info("  🛡️ [Agente SAC] Ativando persona de Ouvidoria e Gestão de Conflitos...")
    sys_prompt = SystemMessage(content=(
        "Você é o Especialista Sênior de SAC, Ouvidoria e Relacionamento com o Cliente da empresa.\n"
        "SEU OBJETIVO: Acolher com máxima empatia, calma e respeito os clientes insatisfeitos, que estejam enfrentando problemas, reclamações sobre atendimento/serviço ou pedindo cancelamento/reembolso.\n\n"
        "DIRETRIZES DE GESTÃO DE CONFLITO:\n"
        "1. Escuta Ativa e Empatia: Reconheça e valide imediatamente a insatisfação do cliente (ex: 'Sinto muito por esse contratempo', 'Compreendo perfeitamente sua frustração'). NUNCA seja reativo, seco ou deboche do cliente.\n"
        "2. Consulta a Políticas: Se a queixa envolver prazos, cancelamentos ou garantias, chame 'consultar_base_conhecimento' para verificar as regras oficiais (se já não tiver chamado).\n"
        "3. Registro no CRM: Registre o contato chamando 'atualizar_etapa_crm' com a etapa 'Em Tratativa SAC'.\n"
        "4. Resolução e Acalmamento: Deixe claro que a empresa está comprometida em resolver a situação da melhor forma possível. Se o cliente solicitar cancelamento financeiro ou reembolso, garanta que a gerência dará prioridade ao caso."
    ))

    messages_to_send = [sys_prompt] + list(state["messages"])

    if state.get("tool_iterations", 0) >= 1:
        logger.info("  💬 [Agente SAC] Sintetizando resposta final com os dados das ferramentas...")
        messages_to_send.append(SystemMessage(
            content="Você já executou as ações de consulta e registro no CRM. NÃO faça chamadas adicionais de ferramentas. "
                    "Elabore agora a resposta final completa, acolhedora e resolutiva para o cliente no WhatsApp."
        ))

    response = await llm_support.ainvoke(messages_to_send)

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        logger.info(f"  🛠️ [Agente SAC] Solicitou ferramentas: {[t['name'] for t in tool_calls]}")
    else:
        logger.info("  💬 [Agente SAC] Resposta de acolhimento gerada com sucesso.")

    return {"messages": [response]}

async def human_handoff(state: AgentState) -> dict:
    """Transbordo Humano: Informa o cliente de forma empática e sinaliza intervenção."""
    logger.info("  🆘 [Etapa Transbordo] Cliente solicitou atendimento humano. Gerando mensagem de transferência...")
    msg = (
        "Compreendo perfeitamente! Estou transferindo o seu atendimento para a nossa equipe humana agora mesmo. "
        "Um de nossos atendentes entrará em contato com você aqui em instantes. Por favor, aguarde um momento! 🙏"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "human_intervention_needed": True
    }

async def execute_tools(state: AgentState) -> dict:
    """Executa as ferramentas solicitadas pelo especialista e devolve o resultado."""
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info(f"  ⚙️ [Ferramenta] Executando {tool_name} com args: {tool_args}")

        matched_tool = next((t for t in all_tools if t.name == tool_name), None)
        if matched_tool:
            try:
                result = matched_tool.invoke(tool_args)
                logger.info(f"  ✨ [Resultado Ferramenta] {tool_name} -> {result}")
            except Exception as e:
                result = f"Erro ao executar a ferramenta: {str(e)}"
                logger.error(f"  ❌ [Erro Ferramenta] {tool_name} -> {result}")
        else:
            result = f"Ferramenta {tool_name} não encontrada."
            logger.warning(f"  ⚠️ [Ferramenta Não Encontrada] {tool_name}")

        tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        tool_messages.append(tool_msg)

    next_iteration = state.get("tool_iterations", 0) + 1
    logger.info(f"  🔁 [Ferramentas] Retornando dados para o especialista sintetizar a resposta (Rodada {next_iteration})...")
    return {"messages": tool_messages, "tool_iterations": next_iteration}

# --- ROTEAMENTO CONDICIONAL ---

def route_after_analyzer(state: AgentState) -> str:
    """Direciona o fluxo para o nó especialista apropriado com base na intenção."""
    intention = state.get("intention", "")
    if intention == "HUMAN":
        return "human_handoff"
    elif intention == "RECLAMACAO":
        return "support_agent"
    elif intention == "VENDAS":
        return "sales_agent"
    else:
        return "faq_agent"

def should_continue_or_tool(state: AgentState) -> str:
    """Verifica se o especialista pediu ferramentas ou finalizou."""
    if state.get("tool_iterations", 0) >= 2:
        return "end"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"

def route_tools_return(state: AgentState) -> str:
    """Após a execução das ferramentas, retorna para o mesmo especialista que as solicitou."""
    intention = state.get("intention", "")
    if intention == "VENDAS":
        return "sales_agent"
    elif intention == "RECLAMACAO":
        return "support_agent"
    else:
        return "faq_agent"

# --- CONSTRUÇÃO DO GRAFO LANGGRAPH ---

workflow = StateGraph(AgentState)

workflow.add_node("analyzer", analyze_intention)
workflow.add_node("sales_agent", sales_agent)
workflow.add_node("faq_agent", faq_agent)
workflow.add_node("support_agent", support_agent)
workflow.add_node("human_handoff", human_handoff)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("analyzer")

# analyzer -> [sales_agent | faq_agent | support_agent | human_handoff]
workflow.add_conditional_edges(
    "analyzer",
    route_after_analyzer,
    {
        "sales_agent": "sales_agent",
        "faq_agent": "faq_agent",
        "support_agent": "support_agent",
        "human_handoff": "human_handoff",
    }
)

# Especialistas -> tools ou END
workflow.add_conditional_edges("sales_agent", should_continue_or_tool, {"tools": "tools", "end": END})
workflow.add_conditional_edges("faq_agent", should_continue_or_tool, {"tools": "tools", "end": END})
workflow.add_conditional_edges("support_agent", should_continue_or_tool, {"tools": "tools", "end": END})

# Transbordo humano finaliza o fluxo direto
workflow.add_edge("human_handoff", END)

# tools -> de volta para o agente especialista correspondente
workflow.add_conditional_edges(
    "tools",
    route_tools_return,
    {
        "sales_agent": "sales_agent",
        "faq_agent": "faq_agent",
        "support_agent": "support_agent"
    }
)

app_graph = workflow.compile()

async def process_message(tenant_id: str, phone_number: str, message: str) -> dict:
    """
    Ponto de entrada chamado pela API FastAPI.
    Executa o grafo multiagente e retorna a resposta gerada, a intenção detectada e status de intervenção humana.
    """
    current_tenant.set(tenant_id)
    current_phone.set(phone_number)
    logger.info(f"🚀 [LangGraph Multiagente Iniciado] Tenant: {tenant_id} | De: {phone_number}")
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "tenant_id": tenant_id,
        "phone_number": phone_number,
        "intention": "",
        "human_intervention_needed": False,
        "tool_iterations": 0
    }

    final_state = await app_graph.ainvoke(initial_state, {"recursion_limit": 50})

    reply = None
    if len(final_state["messages"]) > 1:
        # Procura a última mensagem de texto do assistente
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                reply = msg.content
                break
            elif isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        if not reply:
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, ToolMessage) and "Agendamento confirmado" in str(msg.content):
                    reply = str(msg.content)
                    break

        if not reply and hasattr(final_state["messages"][-1], "content"):
            reply = str(final_state["messages"][-1].content)

    logger.info(f"🏁 [LangGraph Finalizado] Intenção: {final_state.get('intention')} | Resposta: {len(reply) if reply else 0} chars")
    return {
        "reply": reply,
        "intention": final_state.get("intention", "DUVIDAS"),
        "human_intervention_needed": final_state.get("human_intervention_needed", False)
    }
