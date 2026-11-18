"""Generate user-manual illustration candidates with PaperVizAgent (Apache-2.0).

PaperVizAgent (google-research/papervizagent) runs a Retriever -> Planner ->
Stylist -> Visualizer -> Critic pipeline on Gemini models through Vertex AI.
This driver feeds it one spec per figure: the text of the manual subsection the
figure illustrates, plus a hand-written visual brief and caption
(``user_manual_figures/section*/specs/*.json``). It writes every pipeline stage
image and the agents' text descriptions so a human can pick and review the
final candidates. Run it on Cloud Batch (see ``run_section.sh``), not locally.

The images are illustrations, not data: no number in them may come from the
generator. Every label is checked against the brief before a figure is used.
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

TEXT_MODELS = ["gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro"]
IMAGE_MODELS = [
    "gemini-3-pro-image",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-flash-image-preview",
    "gemini-2.5-flash-image",
]


def subsection_text(manual: Path, number: str) -> str:
    """Return the markdown of subsection ``### <number> ...`` up to the next heading."""
    text = manual.read_text(encoding="utf-8")
    m = re.search(rf"^### {re.escape(number)} .*$", text, flags=re.M)
    if m is None:
        raise KeyError(f"subsection {number} not found in {manual}")
    nxt = re.search(r"^(### |## )", text[m.end() :], flags=re.M)
    end = m.end() + nxt.start() if nxt else len(text)
    return text[m.start() : end].strip()


def probe(client, candidates, image):
    """Return the first model id that answers a minimal request."""
    from google.genai import types

    for name in candidates:
        try:
            cfg = (
                types.GenerateContentConfig(response_modalities=["IMAGE"])
                if image
                else None
            )
            prompt = "a small blue square on white" if image else "Reply with OK."
            client.models.generate_content(model=name, contents=prompt, config=cfg)
            print(f"model probe ok: {name}", flush=True)
            return name
        except Exception as exc:  # noqa: BLE001 - report and try the next id
            print(f"model probe failed: {name}: {str(exc)[:200]}", flush=True)
    raise RuntimeError(f"no usable model among {candidates}")


def save_images(result: dict, dest: Path) -> list:
    """Write every base64 stage image in ``result`` to ``dest``; return names."""
    names = []
    for key, val in result.items():
        if key.endswith("_base64_jpg") and val:
            raw = val.split(",", 1)[1] if val.startswith("data:") else val
            name = key.replace("target_diagram_", "").replace("_base64_jpg", "")
            path = dest / f"{name}.jpg"
            path.write_bytes(base64.b64decode(raw))
            names.append(path.name)
    return sorted(names)


async def run(args):
    pva = Path(args.pva_dir).resolve()
    sys.path.insert(0, str(pva))
    os.chdir(pva)
    from google import genai

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    client = genai.Client(vertexai=True, project=project, location=location)
    text_model = args.text_model or probe(client, TEXT_MODELS, image=False)
    image_model = args.image_model or probe(client, IMAGE_MODELS, image=True)

    from agents.critic_agent import CriticAgent
    from agents.planner_agent import PlannerAgent
    from agents.polish_agent import PolishAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.stylist_agent import StylistAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.visualizer_agent import VisualizerAgent
    from utils import config
    from utils.paperviz_processor import PaperVizProcessor

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    manual = Path(args.manual).resolve()
    out = Path(args.out).resolve()
    figures = [f for f in spec["figures"] if not args.only or f["id"] in args.only]

    inputs = []
    for fig in figures:
        body = subsection_text(manual, fig["subsection"])
        for c in range(args.candidates):
            inputs.append(
                {
                    "filename": f"{fig['id']}__c{c}",
                    "candidate_id": c,
                    "caption": fig["caption"],
                    "content": body,
                    "visual_intent": f"{spec['style']}\n\n{fig['visual_intent']}",
                    "additional_info": {"rounded_ratio": fig["aspect_ratio"]},
                    "max_critic_rounds": args.critic_rounds,
                }
            )

    exp = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=args.mode,
        retrieval_setting=args.retrieval,
        model_name=text_model,
        image_model_name=image_model,
        work_dir=pva,
    )
    processor = PaperVizProcessor(
        exp_config=exp,
        vanilla_agent=VanillaAgent(exp_config=exp),
        planner_agent=PlannerAgent(exp_config=exp),
        visualizer_agent=VisualizerAgent(exp_config=exp),
        stylist_agent=StylistAgent(exp_config=exp),
        critic_agent=CriticAgent(exp_config=exp),
        retriever_agent=RetrieverAgent(exp_config=exp),
        polish_agent=PolishAgent(exp_config=exp),
    )

    manifest = {
        "text_model": text_model,
        "image_model": image_model,
        "mode": args.mode,
        "retrieval": args.retrieval,
        "critic_rounds": args.critic_rounds,
        "candidates": {},
    }
    async for res in processor.process_queries_batch(
        inputs, max_concurrent=args.concurrency, do_eval=False
    ):
        fig_id, cand = res["filename"].split("__")
        dest = out / fig_id / cand
        dest.mkdir(parents=True, exist_ok=True)
        images = save_images(res, dest)
        text = {
            k: v
            for k, v in res.items()
            if not k.endswith("_base64_jpg") and isinstance(v, (str, int, float, list))
            and k not in ("content",)
        }
        (dest / "pipeline.json").write_text(json.dumps(text, indent=1, default=str))
        manifest["candidates"][res["filename"]] = images
        print(f"done {res['filename']}: {len(images)} images", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--spec", required=True)
    p.add_argument("--manual", default="user_manual.md")
    p.add_argument("--pva-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--critic-rounds", type=int, default=3)
    p.add_argument("--mode", default="demo_full")
    p.add_argument("--retrieval", default="auto")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--text-model", default="")
    p.add_argument("--image-model", default="")
    p.add_argument("--only", nargs="*", default=[])
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
