import asyncio
import os
import sys

# Garante suporte a UTF-8 no terminal Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Garante importação do agent
from agent import process_message

async def run_tests():
    test_cases = [
        {
            "name": "Cenário 1: VENDAS (Interesse em agendar / valores)",
            "message": "Olá! Gostaria de saber o valor da limpeza de pele e se vocês têm horário disponível para amanhã à tarde?",
            "expected_intent": "VENDAS"
        },
        {
            "name": "Cenário 2: DÚVIDAS (FAQ / Localização)",
            "message": "Onde fica a clínica de vocês e qual o horário de funcionamento?",
            "expected_intent": "DUVIDAS"
        },
        {
            "name": "Cenário 3: RECLAMAÇÃO (SAC / Insatisfação)",
            "message": "Estou muito insatisfeito com o procedimento feito na semana passada, ficou péssimo e quero reembolso!",
            "expected_intent": "RECLAMACAO"
        },
        {
            "name": "Cenário 4: HUMANO (Transbordo)",
            "message": "Por favor, me transfira para um atendente humano agora.",
            "expected_intent": "HUMAN"
        }
    ]

    print("\n" + "="*70)
    print("🧪 INICIANDO BATERIA DE TESTES MULTIAGENTE (LangGraph)")
    print("="*70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testando: {test['name']}")
        print(f"💬 Mensagem enviada: \"{test['message']}\"")
        try:
            result = await process_message(
                tenant_id="test-tenant",
                phone_number="5511999999999",
                message=test["message"]
            )
            print(f"🎯 Intenção Classificada: {result.get('intention')} (Esperado: {test['expected_intent']})")
            print(f"🤖 Resposta ({len(result.get('reply') or '')} chars):")
            print(f"   \"{result.get('reply')}\"")
            print(f"🆘 Intervenção Humana: {result.get('human_intervention_needed')}")
            
            # Verificação básica
            if result.get("intention") == test["expected_intent"]:
                print("✅ [PASS] Intenção correta!")
            else:
                print(f"⚠️ [WARN] Intenção diverge: obteve {result.get('intention')}, esperava {test['expected_intent']}")
        except Exception as e:
            print(f"❌ [ERRO]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("🏁 TESTES FINALIZADOS")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(run_tests())
