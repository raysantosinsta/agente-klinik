from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agent import process_message
import logging
import logging_loki
from prometheus_fastapi_instrumentator import Instrumentator

logging_loki.emitter.LokiEmitter.level_tag = "level"
loki_handler = logging_loki.LokiHandler(
    url="http://127.0.0.1:3100/loki/api/v1/push",
    tags={"app": "python-ai-agent"},
    version="1",
)
logger = logging.getLogger("ai_agent")
logger.setLevel(logging.INFO)
logger.addHandler(loki_handler)

app = FastAPI(title="Klinik AI Agent", description="Agente autônomo LangGraph para atendimento")
Instrumentator().instrument(app).expose(app)

class MessagePayload(BaseModel):
    tenant_id: str
    phone_number: str
    message: str
    
class WebhookResponse(BaseModel):
    status: str
    reply: Optional[str] = None
    human_intervention_needed: bool = False

@app.post("/webhook", response_model=WebhookResponse)
async def handle_message(payload: MessagePayload):
    """
    Recebe a mensagem repassada pelo NestJS, processa no LangGraph e retorna a ação.
    """
    logger.info(f"Mensagem recebida do telefone {payload.phone_number}: {payload.message}")
    try:
        result = await process_message(
            tenant_id=payload.tenant_id,
            phone_number=payload.phone_number,
            message=payload.message
        )
        logger.info("Mensagem processada com sucesso pelo LangGraph.")
        return WebhookResponse(
            status="success",
            reply=result.get("reply"),
            human_intervention_needed=result.get("human_intervention_needed", False)
        )
    except Exception as e:
        logger.error(f"Erro ao processar mensagem no LangGraph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
