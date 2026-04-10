import asyncio
import os
import yaml
from llmrouter.settings import RouterConfig, ConfigStore
from llmrouter.requests import normalize_openai_chat
from llmrouter.services import RouterService, LMStudioClient

async def run_live_test():
    # 1. Konfiguration laden
    config_path = "config/router_config.yaml"
    if not os.path.exists(config_path):
        print(f"Fehler: {config_path} nicht gefunden.")
        return

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    cfg = RouterConfig(**config_data)
    
    # Mock ConfigStore, der die echte Config zurueckgibt
    class SimpleConfigStore:
        def get_config(self):
            return cfg
    
    config_store = SimpleConfigStore()
    
    # 2. LMClient initialisieren (echt)
    lm_client = LMStudioClient()
    
    # 3. RouterService initialisieren
    service = RouterService(config_store, lm_client=lm_client)
    
    # 4. Testfaelle definieren
    test_cases = [
        {
            "name": "Websearch (sollte zu deep gehen)",
            "prompt": "Führe eine websearch nach den aktuellen Goldpreisen durch.",
            "expected_alias": "deep"
        },
        {
            "name": "Commit Message (sollte zu small gehen)",
            "prompt": "Schreibe eine Git-Commit-Nachricht für: Refactoring der Datenbankverbindung.",
            "expected_alias": "small"
        },
        {
            "name": "Komplexe Programmierung (sollte via Judge zu large gehen)",
            "prompt": "Schreibe eine komplexe Python-Klasse für ein neuronales Netz von Grund auf, inklusive Backpropagation.",
            "expected_alias": "large"
        }
    ]
    
    print("Starte Live-LLM-Routing-Test (ohne Mocks)...\n")
    
    # Umgebungsvariablen prüfen
    print(f"DEEP_ENABLED: {os.environ.get('DEEP_ENABLED', 'not set')}")
    print(f"DEEP_API_KEY: {'set' if os.environ.get('DEEP_API_KEY') else 'not set'}")
    print(f"OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'not set'}")
    print("-" * 40)

    for case in test_cases:
        print(f"Testfall: {case['name']}")
        print(f"Prompt: {case['prompt']}")
        
        req = normalize_openai_chat({
            "model": "borg-cpu",
            "messages": [{"role": "user", "content": case["prompt"]}]
        })
        
        # Routing-Entscheidung treffen
        try:
            decision = await service.choose_route(cfg, req)
            print(f"  Routing-Entscheidung: {decision.selected_alias}")
            print(f"  Grund: {decision.reason}")
            
            # Verifikation der Entscheidung
            if decision.selected_alias == case["expected_alias"]:
                print(f"  VERIFIKATION (Entscheidung): PASSED")
            else:
                print(f"  VERIFIKATION (Entscheidung): FAILED (Erwartet {case['expected_alias']})")
            
            # Echten Request an das gewaehlte LLM senden
            print(f"  Sende echten Request an {decision.selected_alias}...")
            
            # Wir nutzen die handle_openai_chat Methode
            payload = {
                "model": "borg-cpu",
                "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": 50
            }
            
            result = await service.handle_openai_chat(payload)
            
            # handle_openai_chat gibt ein Tuple oder direkt die Response zurueck?
            # In services.py: return response_payload (dict)
            # Aber wir sahen ein Tuple im Log: (RouteDecision, final_alias, fallback, response_dict)
            
            if isinstance(result, tuple):
                decision_obj, final_alias, fallback, response = result
            else:
                response = result
            
            if "choices" in response:
                choice = response["choices"][0]
                content = choice["message"]["content"]
                returned_model = response.get("model")
                print(f"  Antwort erhalten von Modell (alias): {final_alias}")
                print(f"  Modell-ID in Response: {returned_model}")
                # print(f"  Antwort-Vorschau: {content[:100]}...")
                
                # Verifikation der Live-Antwort
                if final_alias == case["expected_alias"]:
                    print(f"  VERIFIKATION (Live-Antwort): PASSED")
                elif fallback:
                    print(f"  VERIFIKATION (Live-Antwort): WARNING (Fallback auf {final_alias} erfolgt)")
                    print(f"  Grund fuer Fallback: {decision.reason}")
                elif (case["expected_alias"] == "large" and final_alias == "small" and "judge_unavailable" in decision.reason):
                    print(f"  VERIFIKATION (Live-Antwort): PASSED (Judge-Fallback auf small)")
                else:
                    print(f"  VERIFIKATION (Live-Antwort): FAILED (Erwartet {case['expected_alias']}, bekommen {final_alias})")
            else:
                print(f"  VERIFIKATION (Live-Antwort): FAILED (Keine 'choices' in Antwort)")
                print(f"  Response: {response}")
                
        except Exception as e:
            print(f"  FEHLER während des Tests: {str(e)}")
            import traceback
            traceback.print_exc()
            
        print("-" * 40)

if __name__ == "__main__":
    # Falls DEEP_ENABLED nicht gesetzt ist, fuer den Test setzen
    if "DEEP_ENABLED" not in os.environ:
        os.environ["DEEP_ENABLED"] = "true"
    
    asyncio.run(run_live_test())
