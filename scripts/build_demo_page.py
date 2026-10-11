"""Build the GitHub Pages demo (docs/index.html) from examples/ and results/.

    python scripts/build_demo_page.py

Audio is re-encoded to 16-bit PCM WAV so every browser can play it.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
OUT = ROOT / "docs"
ASSETS = OUT / "demo"
TRACKS = [("mixture", "Original"), ("reference", "Ground truth (target removed)"),
          ("baseline", "Baseline: global removal"), ("sonic", "SONIC: localized removal"),
          ("oracle", "Oracle mask")]
SCENARIO_TEXT = {
    "dog_env": "Dog barking over an environmental background with speech",
    "short_event": "A single short event over a continuous background",
    "intermittent": "A sound occurring at multiple timestamps (SONIC's failure case: it misses part of an event)",
}


def fmt(v, nd=1):
    return "–" if v is None or v != v else f"{v:.{nd}f}"


def example_block(d: Path) -> str:
    meta = json.loads((d / "metrics.json").read_text())
    name = d.name
    scenario = name.split("_", 1)[1]
    dst = ASSETS / name
    dst.mkdir(parents=True, exist_ok=True)
    players = []
    for key, label in TRACKS:
        x, sr = sf.read(d / f"{key}.wav", dtype="float32")
        sf.write(dst / f"{key}.wav", x, sr, subtype="PCM_16")
        players.append(f'<div class="track"><span>{label}</span>'
                       f'<audio controls preload="none" src="demo/{name}/{key}.wav"></audio></div>')
    shutil.copy(d / "localization.png", dst / "localization.png")
    m = meta["metrics"]
    rows = "".join(
        f"<tr><td>{lbl}</td><td>{fmt(m[k].get('sdr_bg'))}</td><td>{fmt(m[k].get('sdr_bg_out'))}</td>"
        f"<td>{fmt(m[k].get('target_suppression_db'))}</td><td>{fmt(m[k].get('loc_iou'), 2)}</td></tr>"
        for k, lbl in [("baseline", "Baseline"), ("sonic", "SONIC"), ("oracle", "Oracle")])
    return f"""
<section class="example">
  <h2>{html.escape(name.split('_')[0].replace('example', 'Example '))}: {html.escape(SCENARIO_TEXT.get(scenario, scenario))}</h2>
  <p class="instr">“{html.escape(meta['instruction'])}”</p>
  <div class="players">{''.join(players)}</div>
  <img src="demo/{name}/localization.png" alt="Spectrogram, CLAP score, and detected vs. ground-truth intervals">
  <div class="scroll"><table>
    <tr><th>Method</th><th>SDR (dB)</th><th>Background SDR outside events (dB)</th><th>Target suppression (dB)</th><th>Localization IoU</th></tr>
    {rows}
  </table></div>
</section>"""


GEN_TRACKS = [("mixture", "Original"), ("gen_global", "Diffusion: global regeneration"),
              ("gen_sonic", "Diffusion: SONIC-masked regeneration"), ("gen_oracle", "Diffusion: oracle-masked"),
              ("sonic", "Subtractive SONIC (reference)")]


def generative_section() -> str:
    src = EXAMPLES / "generative"
    if not src.is_dir():
        return ""
    s = pd.read_csv(ROOT / "results/generative/eval/summary_overall.csv").set_index("method")
    cell = lambda m, k: f"{s.loc[m, k + '_mean']:.1f} <small>[{s.loc[m, k + '_ci_low']:.1f}, {s.loc[m, k + '_ci_high']:.1f}]</small>"
    rows = "".join(
        f"<tr><td>{lbl}</td><td>{cell(m, 'sdr_bg')}</td><td>{cell(m, 'lsd_out')}</td><td>{cell(m, 'lsd_in')}</td></tr>"
        for m, lbl in [("gen_global@0.6", "Global regeneration"), ("gen_sonic@0.6", "<b>SONIC-masked regeneration</b>"),
                       ("gen_oracle@0.6", "Oracle-masked regeneration"), ("sonic", "<i>Subtractive SONIC</i>")])
    blocks = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        dst = ASSETS / "generative" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        players = []
        for key, label in GEN_TRACKS:
            shutil.copy(d / f"{key}.wav", dst / f"{key}.wav")
            players.append(f'<div class="track"><span>{label}</span>'
                           f'<audio controls preload="none" src="demo/generative/{d.name}/{key}.wav"></audio></div>')
        scenario = d.name.rsplit("_", 1)[0]
        blocks.append(f'<h3>{html.escape(SCENARIO_TEXT.get(scenario, scenario).split(" (")[0])}</h3>'
                      f'<div class="players">{"".join(players)}</div>')
    return f"""
<section class="card">
  <h2>Experiment C: the same masks driving a diffusion model</h2>
  <p>A pretrained text-to-audio diffusion model (AudioLDM2) removes the target by <i>regenerating</i> audio, with the target as negative prompt (masked SDEdit, strength 0.6). Without a mask it rewrites the whole clip; with SONIC's mask only the detected regions change and the rest stays bit-identical.</p>
  <div class="scroll"><table>
  <tr><th>Method</th><th>Whole-clip SDR (dB)</th><th>Background LSD outside events (dB, lower is better)</th><th>Background LSD inside events (dB)</th></tr>
  {rows}</table></div>
  <ul>
  <li>Localization improves the generative editor on <b>60/60 clips</b> (p&nbsp;=&nbsp;1.6e-11).</li>
  <li>Off-the-shelf regeneration is far less faithful than subtraction: it invents a plausible but different background (vocoder floor alone is about 7&nbsp;dB LSD). No listening test was run.</li>
  </ul>
  {"".join(blocks)}
  <p><small>AudioLDM2: CC BY-NC-SA 4.0. First eval clip of each scenario, not hand-picked.</small></p>
</section>"""


def summary_table() -> str:
    s = pd.read_csv(ROOT / "results/main/eval/summary_overall.csv").set_index("method")
    cell = lambda m, k: f"{s.loc[m, k + '_mean']:.1f} <small>[{s.loc[m, k + '_ci_low']:.1f}, {s.loc[m, k + '_ci_high']:.1f}]</small>"
    rows = "".join(
        f"<tr><td>{lbl}</td><td>{cell(m, 'sdr_bg')}</td><td>{cell(m, 'sdr_bg_out')}</td><td>{cell(m, 'target_suppression_db')}</td></tr>"
        for m, lbl in [("baseline", "Baseline (global removal)"), ("sonic", "<b>SONIC</b>"), ("oracle", "Oracle mask")])
    return f"""<div class="scroll"><table>
<tr><th>Method</th><th>Whole-clip SDR</th><th>Background SDR outside events</th><th>Target suppression</th></tr>
{rows}</table></div>"""


def main():
    examples = sorted(p for p in EXAMPLES.glob("example*") if p.is_dir())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SONIC: Temporal Control for Audio Editing</title>
<style>
:root {{ --fg:#1d1d1f; --muted:#6e6e73; --bg:#fbfbfd; --card:#fff; --line:#e5e5ea; --accent:#1f6feb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#f5f5f7; --muted:#a1a1a6; --bg:#111113; --card:#1c1c1f; --line:#2c2c30; --accent:#58a6ff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; color:var(--fg); background:var(--bg); }}
main {{ max-width:980px; margin:0 auto; padding:40px 18px 80px; }}
h1 {{ font-size:2rem; margin:0 0 .2em; }} h2 {{ font-size:1.25rem; margin:0 0 .4em; }} h3 {{ font-size:1.02rem; margin:1.2em 0 .5em; }}
.sub {{ color:var(--muted); margin:0 0 1.2em; font-size:1.05rem; }}
a {{ color:var(--accent); }}
.links a {{ margin-right:1.2em; font-weight:600; }}
.card, .example {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:22px; margin:22px 0; }}
.instr {{ font-size:1.1rem; font-style:italic; margin:.2em 0 1em; }}
.players {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:10px 18px; margin-bottom:16px; }}
.track span {{ display:block; font-size:.85rem; color:var(--muted); margin-bottom:2px; }}
audio {{ width:100%; height:36px; }}
img {{ width:100%; border-radius:8px; background:#fff; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; margin-top:12px; font-size:.92rem; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; }}
small {{ color:var(--muted); }}
ul {{ padding-left:1.2em; }}
</style></head><body><main>
<h1>SONIC</h1>
<p class="sub">Semantic and Temporal Control for instruction-guided audio editing: remove a sound described in natural language, <b>only where it occurs</b>.</p>
<p class="links"><a href="https://github.com/yahiag04/SONIC">Code</a><a href="https://github.com/yahiag04/SONIC/blob/main/reports/research_note.md">Research note</a><a href="https://colab.research.google.com/github/yahiag04/SONIC/blob/main/notebooks/sonic_colab.ipynb">Colab</a></p>

<div class="card">
<h2>How it works</h2>
<p>A pretrained text-queried separator (AudioSep) estimates the target sound. SONIC scores 1&nbsp;s windows with CLAP text–audio similarity to find <i>when</i> the target occurs, and subtracts the estimate only inside the detected intervals, with smooth crossfades. Audio outside those intervals is left bit-identical. All methods below share the same separator output; only the edit mask differs.</p>
<h2>Results on 60 held-out mixtures</h2>
{summary_table()}
<ul>
<li>Localized editing preserves the background far better (+16&nbsp;dB outside events) and improves whole-clip SDR by 2.1&nbsp;dB (p&nbsp;=&nbsp;0.007).</li>
<li>Missed event segments cost 2.6&nbsp;dB of target suppression on average, mostly on repeated events.</li>
<li>With ground-truth intervals (oracle), gains reach +5.4&nbsp;dB with no suppression loss, so <b>localization is the bottleneck</b>.</li>
</ul>
<p><small>Mean [95% bootstrap CI]. Synthetic mixtures from ESC-50 (CC BY-NC 3.0) and LibriSpeech (CC BY 4.0). Examples are the first eval clip of each scenario, not hand-picked.</small></p>
</div>
{''.join(example_block(d) for d in examples)}
{generative_section()}
</main></body></html>
"""
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(page)
    (OUT / ".nojekyll").write_text("")
    print(f"wrote {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
