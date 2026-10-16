from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agent import process_message
import os
import logging
import sys

# Configuração do Logger para Console (Terminal)
logger = logging.getLogger("ai_agent")
logger.setLevel(logging.INFO)
logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('\033[96m%(asctime)s\033[0m [%(levelname)s] \033[93m%(name)s\033[0m: %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler opcional para Loki
if os.getenv("LOKI_ENABLED") == "true":
    try:
        import logging_loki
        logging_loki.emitter.LokiEmitter.level_tag = "level"
        loki_handler = logging_loki.LokiHandler(
            url=os.getenv("LOKI_URL", "http://127.0.0.1:3100/loki/api/v1/push"),
            tags={"app": "python-ai-agent"},
            version="1",
        )
        logger.addHandler(loki_handler)
    except Exception as e:
        logger.warning(f"Loki não conectado: {e}")

from prometheus_fastapi_instrumentator import Instrumentator

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
    print("\n" + "="*50)
    logger.info(f"📩 [NOVA MENSAGEM RECEBIDA]")
    logger.info(f"   👤 Telefone: {payload.phone_number}")
    logger.info(f"   🏢 Tenant/Instância: {payload.tenant_id}")
    logger.info(f"   💬 Mensagem: \"{payload.message}\"")
    print("="*50)

    try:
        result = await process_message(
            tenant_id=payload.tenant_id,
            phone_number=payload.phone_number,
            message=payload.message
        )
        logger.info(f"✅ [PROCESSAMENTO CONCLUÍDO COM SUCESSO]")
        logger.info(f"   🤖 Resposta gerada: \"{result.get('reply')}\"")
        logger.info(f"   🆘 Intervenção humana necessária: {result.get('human_intervention_needed', False)}")
        print("="*50 + "\n")

        return WebhookResponse(
            status="success",
            reply=result.get("reply"),
            human_intervention_needed=result.get("human_intervention_needed", False)
        )
    except Exception as e:
        logger.error(f"❌ [ERRO NO PROCESSAMENTO]: {str(e)}", exc_info=True)
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
