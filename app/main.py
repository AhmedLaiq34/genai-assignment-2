"""Local browser demo for evidence-linked trial-criteria screening."""
from __future__ import annotations

import html
import json
import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
import yaml

from src.clinical_match.attention import dense_attention, tiled_attention
from src.clinical_match.decoding import (
    cached_explicit_greedy,
    cached_library_greedy,
    speculative_greedy,
)
from src.clinical_match.evaluation import hash_directory, load_eval_model
from src.clinical_match.screening import ScreeningService, aggregate_results

CONFIG_PATH = ROOT / "configs/eval.yaml"
ADAPTER_PATH = ROOT / "artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/selected_adapter"
_lock = threading.RLock()
_runtime: dict[str, Any] = {"tokenizer": None, "model": None, "draft": None, "cfg": None, "device": None}
_export_dir = tempfile.TemporaryDirectory(prefix="clinical-screening-")


def split_note(text: str) -> list[dict[str, Any]]:
    """Split note into stable, zero-based sentence IDs for model citation."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a patient note first.")
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if p.strip()]
    if not pieces:
        raise ValueError("Enter at least one nonempty sentence.")
    return [{"id": i, "text": p} for i, p in enumerate(pieces)]


def parse_criteria(rows: Any) -> list[tuple[str, str]]:
    if rows is None:
        raise ValueError("Add at least one criterion.")
    if hasattr(rows, "to_dict"):
        rows = rows.values.tolist()
    criteria = []
    for row in rows:
        if not row or len(row) < 2:
            continue
        kind, text = str(row[0] or "").strip().lower(), str(row[1] or "").strip()
        if not text:
            continue
        if kind not in {"inclusion", "exclusion"}:
            raise ValueError("Each criterion type must be inclusion or exclusion.")
        criteria.append((kind, text))
    if not criteria:
        raise ValueError("Add at least one criterion with text.")
    return criteria


def ensure_runtime() -> dict[str, Any]:
    with _lock:
        if _runtime["model"] is not None:
            return _runtime
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(cfg["device"])
        adapter = ADAPTER_PATH if ADAPTER_PATH.is_dir() else None
        if adapter is not None:
            cfg["adapter_path"] = str(adapter)
            cfg["adapter_sha256"] = hash_directory(adapter)
        tokenizer, model = load_eval_model(cfg, device, adapter_path=adapter)
        _runtime.update(tokenizer=tokenizer, model=model, cfg=cfg, device=device)
        return _runtime


def _get_draft(rt: dict[str, Any]) -> Any:
    with _lock:
        if _runtime["draft"] is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
            draft_cfg = model_cfg["draft"]
            draft_tokenizer = AutoTokenizer.from_pretrained(
                draft_cfg["id"], revision=draft_cfg["revision"], use_fast=True
            )
            if draft_tokenizer.get_vocab() != rt["tokenizer"].get_vocab():
                raise ValueError("Pinned target and draft vocabularies do not match")
            dtype = torch.float16 if rt["device"].type == "cuda" else torch.float32
            draft = AutoModelForCausalLM.from_pretrained(
                draft_cfg["id"], revision=draft_cfg["revision"],
                dtype=dtype, low_cpu_mem_usage=True,
                attn_implementation=rt["cfg"].get("attn_implementation", "sdpa"),
            ).to(rt["device"])
            draft.eval()
            _runtime["draft"] = draft
        return _runtime["draft"]


def _generator(rt: dict[str, Any], mode: str):
    if mode == "Explicit KV cache":
        return cached_explicit_greedy, mode
    if mode == "Speculative (k=4)":
        draft = _get_draft(rt)
        def generate(model, input_ids, **kwargs):
            kwargs.pop("pad_token_id", None)
            return speculative_greedy(model, draft, input_ids, draft_k=4, **kwargs)
        return generate, mode
    return cached_library_greedy, "Library cached greedy"


def screen(note: str, rows: Any, mode: str):
    try:
        sentences, criteria = split_note(note), parse_criteria(rows)
        rt = ensure_runtime()
        generator, mode_name = _generator(rt, mode)
        with _lock:
            service = ScreeningService(
                rt["tokenizer"], rt["model"], rt["cfg"], rt["device"],
                generate_fn=generator,
            )
            results = [service.screen(sentences, kind, text, generation_mode=mode_name)
                       for kind, text in criteria]
        aggregate = aggregate_results(results)
        report = {
            "runtime": {"model_id": rt["cfg"]["model_id"],
                        "model_revision": rt["cfg"]["model_revision"],
                        "adapter_sha256": rt["cfg"].get("adapter_sha256"),
                        "device": str(rt["device"]), "decoder": mode_name},
            "criteria": results, "aggregate": aggregate,
        }
        if mode_name == "Speculative (k=4)":
            model_config = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
            report["runtime"]["draft_model_id"] = model_config["draft"]["id"]
            report["runtime"]["draft_model_revision"] = model_config["draft"]["revision"]
        cards = []
        for item in results:
            excerpts = "".join(f'<blockquote><b>[{e["id"]}]</b> {html.escape(e["text"])}</blockquote>' for e in item.get("evidence_excerpts", []))
            flags = item.get("flags", {})
            flag_text = ", ".join(name.replace("_", " ") for name, active in flags.items() if active) or "none"
            invalid = item.get("invalid_evidence_values", [])
            invalid_text = f" · Invalid evidence IDs: {html.escape(str(invalid))}" if invalid else ""
            cards.append(f"<section class='result'><h3>{html.escape(item.get('criterion_type','').title())}: {html.escape(item.get('criterion_text',''))}</h3><p><b>Prediction:</b> {html.escape(str(item.get('label') or 'Needs review'))} · <b>Parse:</b> {html.escape(str(item.get('parse_status')))} · <b>Evidence:</b> {html.escape(str(item.get('evidence_status','unknown')))}</p>{excerpts}<p>{html.escape(str(item.get('explanation','')))}</p><p><b>Review flags:</b> {html.escape(flag_text)}{invalid_text}</p></section>")
        panel = f"<div class='summary'><h2>{html.escape(aggregate['outcome'].replace('_',' ').title())}</h2><p>{html.escape(aggregate.get('outcome_description',''))}</p><small>{html.escape(aggregate['educational_notice'])}</small></div>" + "".join(cards)
        return panel, report, report, "Screening finished. Nothing was written to disk. Exports include cited text and are created only when you request a download."
    except Exception as exc:  # surfaced in UI so the operator can recover
        return f"<div class='error'>{html.escape(type(exc).__name__ + ': ' + str(exc))}</div>", None, None, "Correct the input or restart the app if model loading failed."


def export_report(report: dict[str, Any] | None, format_name: str):
    if not report:
        raise ValueError("Screen a note before requesting an export.")
    destination = Path(_export_dir.name) / f"screening-{uuid.uuid4().hex}.{format_name.lower()}"
    if format_name == "JSON":
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(destination)
    aggregate = report["aggregate"]
    cards = []
    for item in report["criteria"]:
        snippets = "".join(f"<blockquote>[{e['id']}] {html.escape(e['text'])}</blockquote>" for e in item.get("evidence_excerpts", []))
        flags = ", ".join(k.replace("_", " ") for k, v in item.get("flags", {}).items() if v) or "none"
        cards.append(f"<section class='result'><h2>{html.escape(item['criterion_type'].title())}: {html.escape(item['criterion_text'])}</h2><p><b>Prediction:</b> {html.escape(str(item.get('label') or 'Needs review'))} · <b>Evidence:</b> {html.escape(str(item.get('evidence_status','unknown')))} · <b>Parse:</b> {html.escape(str(item.get('parse_status','unknown')))}</p>{snippets}<p>{html.escape(item.get('explanation',''))}</p><p><b>Review flags:</b> {html.escape(flags)}</p></section>")
    body = f"<div class='summary'><h1>{html.escape(aggregate['outcome'].replace('_',' ').title())}</h1><p>{html.escape(aggregate.get('outcome_description',''))}</p></div>" + "".join(cards)
    document = ("<!doctype html><html><head><meta charset='utf-8'><title>Screening report</title>"
                f"<style>body{{font:16px system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#172033}}{CSS}</style></head><body>"
                "<h1>Clinical trial criteria screening</h1>" + body +
                f"<p>{html.escape(aggregate['educational_notice'])}</p><p>Model: {html.escape(report['runtime']['model_id'])} · Decoder: {html.escape(report['runtime']['decoder'])}</p></body></html>")
    destination.write_text(document, encoding="utf-8")
    return str(destination)


def attention_demo(length: int):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(42)
        q = torch.randn((1, 4, length, 32), generator=generator, device=device)
        k = torch.randn((1, 2, length, 32), generator=generator, device=device)
        v = torch.randn((1, 2, length, 32), generator=generator, device=device)
        outputs, times = [], []
        for fn in (dense_attention, tiled_attention):
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter(); outputs.append(fn(q, k, v))
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return (f"Length {length} Â· max |dense âˆ’ tiled| = "
                f"{(outputs[0].float()-outputs[1].float()).abs().max().item():.6g}\n"
                f"Dense {times[0]:.2f} ms Â· Tiled {times[1]:.2f} ms Â· {device}")
    except Exception as exc:
        return f"Attention comparison failed: {type(exc).__name__}: {exc}"


CSS = """.gradio-container{max-width:1160px!important}.result,.summary{border:1px solid #d7deea;border-radius:12px;padding:16px;margin:12px 0}.summary{background:#eff6ff}.result blockquote{border-left:3px solid #6b8fd6;margin:8px 0;padding:8px 12px;background:#f7f9fc}.error{color:#8b1e1e;background:#fff0f0;padding:14px;border-radius:8px}"""

with gr.Blocks(title="Clinical Trial Criteria Screener") as demo:
    gr.Markdown("# Clinical trial criteria screener\nEducational prototype. Review every cited sentence and prediction; this is not a clinical enrollment decision.")
    gr.Markdown("Results stay in this app session unless you choose an export. Downloaded reports contain criterion text and cited patient excerpts and are held in a temporary directory cleared when the app closes.")
    with gr.Tab("Screen a synthetic or de-identified note"):
        note = gr.Textbox(label="Patient note", lines=8, placeholder="One sentence per line works best.")
        criteria = gr.Dataframe(headers=["Type (inclusion/exclusion)", "Criterion"], datatype=["str", "str"], row_count=(2, "dynamic"), column_count=(2, "fixed"), value=[["inclusion", "Age 18 years or older"], ["exclusion", "History of active malignancy"]], label="Criteria")
        mode = gr.Dropdown(["Library cached greedy", "Explicit KV cache", "Speculative (k=4)"], value="Library cached greedy", label="Decoder")
        run = gr.Button("Screen note", variant="primary")
        status = gr.Markdown()
        result_html = gr.HTML(label="Evidence-linked results")
        raw = gr.JSON(label="Structured result and runtime provenance")
        report_state = gr.State()
        with gr.Row():
            export_json = gr.Button("Create JSON download")
            export_html = gr.Button("Create HTML download")
        json_file = gr.File(label="JSON download")
        html_file = gr.File(label="HTML download")
        run.click(screen, inputs=[note, criteria, mode], outputs=[result_html, raw, report_state, status])
        export_json.click(lambda report: export_report(report, "JSON"), inputs=report_state, outputs=json_file)
        export_html.click(lambda report: export_report(report, "HTML"), inputs=report_state, outputs=html_file)
    with gr.Tab("Attention demonstration"):
        gr.Markdown("Compare the assignment's dense and tiled attention implementations on synthetic tensors. This panel does not use patient data.")
        length = gr.Slider(16, 256, value=64, step=16, label="Sequence length")
        attn_button = gr.Button("Run dense vs tiled comparison")
        attn_result = gr.Textbox(label="Numerical and timing result", lines=3)
        attn_button.click(attention_demo, inputs=length, outputs=attn_result)
    gr.Markdown("Decoder speed depends on GPU and prompt length. Speculative mode also loads the pinned 0.5B draft model and needs more GPU memory; the standard screening workflow uses one model.")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", inbrowser=True, show_error=True,
        theme=gr.themes.Soft(), css=CSS,
    )

