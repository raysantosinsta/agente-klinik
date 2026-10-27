import asyncio
import os
import sys

# Garante suporte a UTF-8 no terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from agent import process_message

async def run_fechamento_test():
    print("\n" + "="*70)
    print("🧪 TESTE: FECHAMENTO DE AGENDAMENTO E DISPARO DE WEBHOOK")
    print("="*70)

    # Mensagem onde o cliente confirma o agendamento
    msg = "Perfeito, pode agendar a Limpeza de Pele para mim amanhã às 14:00! Meu nome é Mariana Silva."
    print(f"💬 Mensagem do Cliente: \"{msg}\"")

    try:
        result = await process_message(
            tenant_id="test-tenant",
            phone_number="5511998877665",
            message=msg
        )
        print("\n🎯 Resultado do Processamento:")
        print(f"   Intenção: {result.get('intention')}")
        print(f"   Resposta da IA:\n   \"{result.get('reply')}\"")
        print(f"   Intervenção Humana: {result.get('human_intervention_needed')}")
        print("\n✅ Fluxo executado com sucesso!")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)

if __name__ == "__main__":
    asyncio.run(run_fechamento_test())
