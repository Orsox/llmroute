import argparse
import json
from typing import Any

import httpx


def _headers(token: str | None) -> dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _print_meta(name: str, response: httpx.Response) -> None:
    print(f"\n=== {name} ===")
    print(f"status: {response.status_code}")
    print("selected:", response.headers.get("x-router-selected-model"))
    print("judge:", response.headers.get("x-router-judge-model"))
    print("reason:", response.headers.get("x-router-reason"))
    print("fallback:", response.headers.get("x-router-fallback"))


def _print_json_preview(data: Any) -> None:
    dumped = json.dumps(data, indent=2, ensure_ascii=False)
    if len(dumped) > 1000:
        dumped = dumped[:1000] + "\n...<truncated>..."
    print(dumped)


def run_non_stream_case(client: httpx.Client, base_url: str, name: str, path: str, payload: dict[str, Any], token: str | None) -> None:
    response = client.post(base_url + path, headers=_headers(token), json=payload)
    _print_meta(name, response)
    try:
        _print_json_preview(response.json())
    except Exception:
        print(response.text[:1000])


def run_stream_case(client: httpx.Client, base_url: str, name: str, path: str, payload: dict[str, Any], token: str | None) -> None:
    with client.stream("POST", base_url + path, headers=_headers(token), json=payload) as response:
        _print_meta(name, response)
        lines_seen = 0
        for line in response.iter_lines():
            if not line:
                continue
            print(line)
            lines_seen += 1
            if lines_seen >= 12:
                print("...<stream preview truncated>...")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo requests for the local LLM router")
    parser.add_argument("--router-url", default="http://127.0.0.1:12345", help="Router base URL")
    parser.add_argument("--token", default=None, help="Optional bearer token")
    parser.add_argument("--model", default="borg-cpu", help="Public router model name (e.g. borg-cpu)")
    args = parser.parse_args()

    base_url = args.router_url.rstrip("/")
    token = args.token
    model = args.model

    with httpx.Client(timeout=120.0) as client:
        short_chat = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Erkläre in 2 Sätzen, was ein HTTP-Proxy ist."}
            ],
            "max_tokens": 120,
            "stream": False,
        }
        run_non_stream_case(client, base_url, "Short Chat (small expected)", "/v1/chat/completions", short_chat, token)

        complex_prompt = "Bitte analysiere die folgenden Anforderungen und erstelle einen präzisen Umsetzungsplan.\n" + ("Kontextblock. " * 2500)
        complex_chat = {
            "model": model,
            "messages": [{"role": "user", "content": complex_prompt}],
            "max_tokens": 2600,
            "stream": False,
        }
        run_non_stream_case(client, base_url, "Complex Chat (large expected)", "/v1/chat/completions", complex_chat, token)

        vision_chat = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Beschreibe kurz dieses Bild."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Yz7L2QAAAAASUVORK5CYII="
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 100,
            "stream": False,
        }
        run_non_stream_case(client, base_url, "Vision Chat", "/v1/chat/completions", vision_chat, token)

        tool_chat = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Welche Wetterdaten brauchst du, um morgen Regen in Berlin abzuschätzen?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "date": {"type": "string"}
                            },
                            "required": ["city", "date"]
                        },
                    },
                }
            ],
            "max_tokens": 300,
            "stream": True,
        }
        run_stream_case(client, base_url, "Tooluse Chat Stream", "/v1/chat/completions", tool_chat, token)

        anthropic_msg = {
            "model": model,
            "max_tokens": 180,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Gib mir drei Ideen für einen robusten Prompt-Router."}
                    ],
                }
            ],
            "stream": False,
        }
        run_non_stream_case(client, base_url, "Anthropic Messages", "/v1/messages", anthropic_msg, token)

        completion_case = {
            "model": model,
            "prompt": "Schreibe drei prägnante Produktnamen für einen lokalen LLM-Router.",
            "max_tokens": 80,
            "stream": False,
        }
        run_non_stream_case(client, base_url, "Legacy Completions", "/v1/completions", completion_case, token)


if __name__ == "__main__":
    main()

