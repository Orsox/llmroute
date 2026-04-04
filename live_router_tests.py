import argparse
import asyncio
import base64
import json
import struct
import textwrap
import zlib
from pathlib import Path
from typing import Any, Optional

import httpx

from llmrouter.app import create_app


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)


def _encode_png_rgb(width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> bytes:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            r, g, b = pixels[y][x]
            raw.extend((r, g, b))

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_chunk(b"IHDR", ihdr))
    png.extend(_chunk(b"IDAT", idat))
    png.extend(_chunk(b"IEND", b""))
    return bytes(png)


def _draw_circle(pixels: list[list[tuple[int, int, int]]], cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    h = len(pixels)
    w = len(pixels[0])
    rr = radius * radius
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        dy = y - cy
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            dx = x - cx
            if dx * dx + dy * dy <= rr:
                pixels[y][x] = color


def _draw_rect(pixels: list[list[tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h = len(pixels)
    w = len(pixels[0])
    for y in range(max(0, y0), min(h, y1)):
        row = pixels[y]
        for x in range(max(0, x0), min(w, x1)):
            row[x] = color


def _draw_line(
    pixels: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for oy in range(-thickness, thickness + 1):
            for ox in range(-thickness, thickness + 1):
                yy = y0 + oy
                xx = x0 + ox
                if 0 <= yy < len(pixels) and 0 <= xx < len(pixels[0]):
                    pixels[yy][xx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def generate_child_with_bouquet_png(path: Path) -> bytes:
    width, height = 640, 420
    sky_top = (189, 226, 255)
    sky_bottom = (239, 248, 255)
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    for y in range(height):
        t = y / max(1, (height - 1))
        r = int(sky_top[0] * (1 - t) + sky_bottom[0] * t)
        g = int(sky_top[1] * (1 - t) + sky_bottom[1] * t)
        b = int(sky_top[2] * (1 - t) + sky_bottom[2] * t)
        for x in range(width):
            pixels[y][x] = (r, g, b)

    _draw_rect(pixels, 0, 300, width, height, (139, 198, 104))
    _draw_circle(pixels, 88, 86, 44, (255, 225, 86))

    _draw_rect(pixels, 280, 215, 360, 325, (47, 110, 220))
    _draw_rect(pixels, 280, 325, 317, 395, (36, 58, 102))
    _draw_rect(pixels, 323, 325, 360, 395, (36, 58, 102))

    _draw_circle(pixels, 320, 155, 50, (247, 214, 183))
    _draw_circle(pixels, 287, 135, 10, (57, 38, 33))
    _draw_circle(pixels, 353, 135, 10, (57, 38, 33))
    _draw_line(pixels, 270, 145, 370, 145, (65, 41, 35), thickness=8)

    _draw_circle(pixels, 302, 160, 4, (30, 30, 30))
    _draw_circle(pixels, 338, 160, 4, (30, 30, 30))
    _draw_line(pixels, 307, 185, 333, 193, (145, 65, 62), thickness=2)

    _draw_rect(pixels, 235, 235, 280, 256, (247, 214, 183))
    _draw_rect(pixels, 360, 235, 424, 256, (247, 214, 183))

    bouquet_center_x, bouquet_center_y = 425, 242
    _draw_line(pixels, 425, 260, 445, 315, (53, 145, 67), thickness=2)
    _draw_line(pixels, 415, 262, 430, 316, (53, 145, 67), thickness=2)
    _draw_line(pixels, 435, 258, 455, 316, (53, 145, 67), thickness=2)

    flower_colors = [
        (248, 97, 119),
        (255, 168, 77),
        (120, 187, 82),
        (160, 108, 255),
        (255, 92, 170),
        (241, 208, 77),
    ]
    flower_positions = [
        (402, 236),
        (422, 222),
        (446, 236),
        (438, 254),
        (410, 255),
        (427, 242),
    ]
    for (fx, fy), color in zip(flower_positions, flower_colors):
        _draw_circle(pixels, fx, fy, 14, color)
        _draw_circle(pixels, fx, fy, 5, (250, 242, 227))

    _draw_rect(pixels, 418, 260, 444, 300, (227, 191, 113))
    _draw_line(pixels, 418, 260, 444, 300, (188, 149, 78), thickness=1)
    _draw_line(pixels, 444, 260, 418, 300, (188, 149, 78), thickness=1)

    png_bytes = _encode_png_rgb(width, height, pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    return png_bytes


def _headers(token: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _extract_openai_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                out.append(str(item.get("text", "")))
        return "".join(out)
    if "text" in first:
        return str(first.get("text") or "")
    return ""


async def _run_requests(client: httpx.AsyncClient, token: Optional[str], image_bytes: bytes, out_dir: Path) -> None:
    health = await client.get("/healthz")
    health.raise_for_status()

    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

    vision_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Bitte beschreibe präzise, was auf dem Bild zu sehen ist (Person, Kleidung, Gegenstand, Farben, Umgebung).",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 350,
        "temperature": 0.2,
        "stream": False,
    }

    vision_resp = await client.post("/v1/chat/completions", headers=_headers(token), json=vision_payload)
    vision_resp.raise_for_status()
    vision_json = vision_resp.json()
    vision_text = _extract_openai_text(vision_json)

    print("\n=== TEST 1: Vision (Kind mit Blumenstrauss) ===")
    print("selected model:", vision_resp.headers.get("x-router-selected-model"))
    print("judge model:", vision_resp.headers.get("x-router-judge-model"))
    print("route reason:", vision_resp.headers.get("x-router-reason"))
    print("fallback:", vision_resp.headers.get("x-router-fallback"))
    print("analysis preview:\n", vision_text[:900])

    story_prompt = textwrap.dedent(
        """
        Schreibe eine kreative, zusammenhaengende und emotional starke Geschichte ueber Baeren.
        Anforderungen:
        - Umfang: exakt vier Seiten Fliesstext-Charakter (ca. 2200-2800 Woerter).
        - Sprache: Deutsch.
        - Stil: literarisch, aber gut lesbar.
        - Struktur: 4 Kapitel mit Ueberschriften (Kapitel 1 bis Kapitel 4), jedes Kapitel soll ungefaehr eine Seite wirken.
        - Inhalt: Charakterentwicklung, Konflikt, Wendepunkt, klares Ende.
        - Keine Aufzaehlungen, kein Meta-Kommentar.
        """
    ).strip()

    story_payload = {
        "model": "qwen/qwen3.5-35b-a3b",
        "messages": [
            {
                "role": "user",
                "content": story_prompt,
            }
        ],
        "max_tokens": 7000,
        "temperature": 0.8,
        "stream": False,
    }

    story_resp = await client.post("/v1/chat/completions", headers=_headers(token), json=story_payload)
    story_resp.raise_for_status()
    story_json = story_resp.json()
    story_text = _extract_openai_text(story_json)

    story_file = out_dir / "baeren_4_seiten_geschichte.txt"
    story_file.write_text(story_text, encoding="utf-8")

    print("\n=== TEST 2: Denkaufgabe grosse LLM (4 Seiten Baeren-Geschichte) ===")
    print("selected model:", story_resp.headers.get("x-router-selected-model"))
    print("judge model:", story_resp.headers.get("x-router-judge-model"))
    print("route reason:", story_resp.headers.get("x-router-reason"))
    print("fallback:", story_resp.headers.get("x-router-fallback"))
    print("story length (chars):", len(story_text))
    print("story length (words):", len(story_text.split()))
    print("story preview:\n", story_text[:900])
    print("story saved:", story_file)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run real LM Studio routing tests")
    parser.add_argument("--router-url", default=None, help="Optional external router URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--config", default="config/router_config.yaml", help="Router config path for in-process mode")
    parser.add_argument("--token", default=None, help="Optional shared bearer token")
    parser.add_argument("--out-dir", default="outputs", help="Directory for generated artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_file = out_dir / "child_with_bouquet.png"
    image_bytes = generate_child_with_bouquet_png(image_file)
    print("generated image:", image_file)

    if args.router_url:
        async with httpx.AsyncClient(base_url=args.router_url.rstrip("/"), timeout=600.0) as client:
            await _run_requests(client, args.token, image_bytes, out_dir)
    else:
        app = create_app(config_path=Path(args.config))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router.local", timeout=600.0) as client:
            await _run_requests(client, args.token, image_bytes, out_dir)


if __name__ == "__main__":
    asyncio.run(main_async())
