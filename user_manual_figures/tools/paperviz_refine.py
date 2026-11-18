"""Apply targeted corrections (or a plain upscale) to picked illustration candidates.

Uses the same Vertex AI image model as PaperVizAgent's refine step (image + edit
prompt -> 2K image). Each item is rendered ``--variants`` times; a human picks the
variant that is correct. Run on Cloud Batch (``run_refine.sh``).
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import types


def render(client, model, prompt, image_bytes, aspect):
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect, image_size="2K"),
    )
    parts = [types.Part.from_text(text=prompt), types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)]
    resp = client.models.generate_content(model=model, contents=parts, config=cfg)
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return part.inline_data.data
    raise RuntimeError("no image in response")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refine", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--model", default="gemini-3-pro-image")
    a = p.parse_args()
    ref = json.loads(Path(a.refine).read_text())
    aspect = {f["id"]: f["aspect_ratio"] for f in json.loads(Path(a.spec).read_text())["figures"]}
    client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global")
    jobs = []
    for it in ref["items"]:
        prompt = ref["upscale_only"] if not it["edit"] else f"{ref['keep']}\n\nChange only this:\n{it['edit']}"
        src = (Path(a.candidates) / it["id"] / it["source"]).read_bytes()
        for v in range(a.variants):
            jobs.append((it["id"], v, prompt, src))

    def run(job):
        fid, v, prompt, src = job
        dest = Path(a.out) / fid
        dest.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                (dest / f"v{v}.png").write_bytes(render(client, a.model, prompt, src, aspect[fid]))
                return f"ok {fid} v{v}"
            except Exception as exc:  # noqa: BLE001
                err = str(exc)[:200]
        return f"FAILED {fid} v{v}: {err}"

    with ThreadPoolExecutor(10) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
