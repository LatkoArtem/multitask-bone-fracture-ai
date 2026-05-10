"""
FracDetect AI — Bone Fracture Detection & Localization
Gradio web interface for the cascade neural pipeline.

Run:
    pip install gradio
    python app.py
"""

import os
import sys
import numpy as np
from PIL import Image

import gradio as gr

# ── resolve paths relative to this file ──────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS  = os.path.join(ROOT, 'notebooks')
sys.path.insert(0, ROOT)

from src.pipeline import CascadePipeline, BONE_CLASSES

# ── auto-detect best available checkpoints ───────────────────────────────
def _find_segmenter() -> str:
    for v in range(5, 1, -1):
        for suffix in ('_best', ''):
            p = os.path.join(NOTEBOOKS, f'fracture_segmenter_v{v}{suffix}.pth')
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        "No segmenter checkpoint found in notebooks/. "
        "Train the model first (run notebook cell 10)."
    )

CLASSIFIER_PATH = os.path.join(NOTEBOOKS, 'multitask_classifier.pth')
SEGMENTER_PATH  = _find_segmenter()

print(f"[FracDetect] Classifier : {os.path.basename(CLASSIFIER_PATH)}")
print(f"[FracDetect] Segmenter  : {os.path.basename(SEGMENTER_PATH)}")

pipeline = CascadePipeline(CLASSIFIER_PATH, SEGMENTER_PATH)

# ── overlay helper ────────────────────────────────────────────────────────
def _apply_mask_overlay(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent red region over fracture pixels."""
    overlay = img_rgb.astype(np.float32).copy()
    red     = np.array([220, 38, 38], dtype=np.float32)       # Tailwind red-600
    alpha   = 0.45

    m = mask.astype(bool)
    overlay[m] = overlay[m] * (1 - alpha) + red * alpha

    # thin bright-red border around the fracture region using dilation
    import cv2
    kernel   = np.ones((5, 5), np.uint8)
    border   = cv2.dilate(mask, kernel, iterations=2) & ~mask
    overlay[border.astype(bool)] = [255, 60, 60]

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ── main inference function ───────────────────────────────────────────────
def analyze(image: Image.Image):
    if image is None:
        return None, _empty_html("Завантажте рентгенівський знімок")

    img_rgb = np.array(image.convert('RGB'))
    result  = pipeline.predict(img_rgb, threshold=0.5)

    # --- result image ---
    if result['is_fractured'] and result['mask'] is not None:
        out_img = _apply_mask_overlay(img_rgb, result['mask'])
    else:
        out_img = img_rgb

    # --- HTML results panel ---
    html = _build_results_html(result)

    return Image.fromarray(out_img), html


# ── HTML builders ─────────────────────────────────────────────────────────
def _empty_html(msg: str) -> str:
    return f"""
    <div style="display:flex;align-items:center;justify-content:center;
                height:200px;color:#64748b;font-size:15px;
                font-family:'Inter',sans-serif;">
        {msg}
    </div>"""


def _bone_confidence_bar(prob: float) -> str:
    """Single confidence bar for the detected bone type."""
    pct = int(prob * 100)
    return f"""
    <div style="margin-top:8px;">
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;color:#64748b;margin-bottom:4px;">
        <span>Впевненість</span><span>{pct}%</span>
      </div>
      <div style="background:#1e293b;border-radius:4px;height:6px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:#38bdf8;
                    border-radius:4px;"></div>
      </div>
    </div>"""


def _build_results_html(r: dict) -> str:
    is_frac   = r['is_fractured']
    frac_pct  = int(r['fracture_prob'] * 100)

    status_bg    = '#450a0a' if is_frac else '#052e16'
    status_bdr   = '#dc2626' if is_frac else '#16a34a'
    status_color = '#fca5a5' if is_frac else '#86efac'
    status_icon  = '⚠' if is_frac else '✓'
    status_text  = 'ПЕРЕЛОМ ВИЯВЛЕНО' if is_frac else 'НОРМА — ПЕРЕЛОМІВ НЕ ВИЯВЛЕНО'

    prob_color = '#dc2626' if frac_pct >= 50 else '#16a34a'
    prob_bg    = '#1c0a0a' if frac_pct >= 50 else '#0a1c10'

    bone_conf = r['bone_probs'].get(r['bone_type'], 0)
    bone_bar  = _bone_confidence_bar(bone_conf)

    return f"""
    <div style="font-family:'Inter',sans-serif;color:#e2e8f0;padding:4px 0;">

      <!-- Status badge -->
      <div style="background:{status_bg};border:1px solid {status_bdr};
                  border-radius:10px;padding:14px 20px;margin-bottom:16px;
                  display:flex;align-items:center;gap:12px;">
        <span style="font-size:28px;color:{status_color};">{status_icon}</span>
        <div>
          <div style="font-size:13px;color:#94a3b8;margin-bottom:2px;">Результат аналізу</div>
          <div style="font-size:16px;font-weight:700;color:{status_color};
                      letter-spacing:.5px;">{status_text}</div>
        </div>
      </div>

      <!-- Fracture probability -->
      <div style="background:{prob_bg};border:1px solid #1e293b;
                  border-radius:10px;padding:14px 20px;margin-bottom:16px;">
        <div style="font-size:12px;color:#94a3b8;margin-bottom:6px;
                    text-transform:uppercase;letter-spacing:.8px;">
          Ймовірність перелому
        </div>
        <div style="display:flex;align-items:center;gap:14px;">
          <div style="font-size:36px;font-weight:800;color:{prob_color};
                      font-variant-numeric:tabular-nums;">{frac_pct}%</div>
          <div style="flex:1;">
            <div style="background:#1e293b;border-radius:6px;height:10px;overflow:hidden;">
              <div style="width:{frac_pct}%;height:100%;background:{prob_color};
                          border-radius:6px;"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Bone type — single label -->
      <div style="background:#0f172a;border:1px solid #1e293b;
                  border-radius:10px;padding:14px 20px;">
        <div style="font-size:12px;color:#94a3b8;margin-bottom:8px;
                    text-transform:uppercase;letter-spacing:.8px;">
          Тип кістки
        </div>
        <div style="font-size:24px;font-weight:700;color:#38bdf8;">
          {r['bone_type']}
        </div>
        {bone_bar}
      </div>

    </div>"""


# ── CSS ───────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, #root {
    background: #020617 !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
}

body, .gradio-container {
    background: #020617 !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    min-height: 100vh !important;
    background: #020617 !important;
}

/* Gradio inner wrappers */
.main, .wrap, .contain,
div.svelte-1gfkfd6,
.app, .page,
.gradio-container > div,
section.svelte-vt1mxs,
div.svelte-vt1mxs {
    background: #020617 !important;
}

/* Gradio default light panel overrides */
:root {
    --body-background-fill: #020617 !important;
    --background-fill-primary: #020617 !important;
    --background-fill-secondary: #020617 !important;
    --panel-background-fill: #020617 !important;
    --block-background-fill: #020617 !important;
    --border-color-primary: #1e293b !important;
}

/* header */
#header {
    text-align: center;
    padding: 32px 0 24px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 28px;
}
#header h1 {
    font-size: 28px;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
#header p {
    font-size: 14px;
    color: #64748b;
    margin: 0;
}
#header .badge {
    display: inline-block;
    background: #0f3460;
    color: #38bdf8;
    border: 1px solid #1d4ed8;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .8px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* panels */
.panel {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
}

/* upload area */
.upload-box .wrap {
    background: #0f172a !important;
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
}
.upload-box .wrap:hover {
    border-color: #14b8a6 !important;
}

/* result image */
#result-img img {
    border-radius: 10px;
    border: 1px solid #1e293b;
}

/* analyze button */
#analyze-btn {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 10px !important;
    padding: 12px !important;
    letter-spacing: .3px !important;
    transition: opacity .2s !important;
}
#analyze-btn:hover { opacity: .88 !important; }

/* footer */
#footer {
    text-align: center;
    padding: 20px 0 8px;
    color: #334155;
    font-size: 12px;
    border-top: 1px solid #1e293b;
    margin-top: 24px;
}

/* hide default gradio branding */
footer { display: none !important; }
"""


# ── Gradio layout ─────────────────────────────────────────────────────────
seg_name = os.path.basename(SEGMENTER_PATH)
cls_name = os.path.basename(CLASSIFIER_PATH)

with gr.Blocks(css=CSS, title="FracDetect AI") as demo:

    gr.HTML(f"""
    <div id="header">
      <div class="badge">AI Medical Imaging</div>
      <h1>🦴 FracDetect AI</h1>
      <p>Система виявлення та локалізації переломів кісток · Cascade Neural Pipeline</p>
      <p style="color:#1e3a5f;font-size:11px;margin-top:6px;">
        Classifier: {cls_name} &nbsp;|&nbsp; Segmenter: {seg_name}
      </p>
    </div>
    """)

    with gr.Row(equal_height=True):
        # ── Left column: upload ───────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML('<div style="font-size:13px;color:#94a3b8;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px;">'
                    'Рентгенівський знімок</div>')

            img_input = gr.Image(
                type='pil',
                label='',
                elem_classes='upload-box',
                show_label=False,
                height=340,
            )

            btn = gr.Button('🔬  Аналізувати знімок', elem_id='analyze-btn')

            gr.HTML("""
            <div style="margin-top:16px;padding:12px;background:#0a0f1e;
                        border:1px solid #1e293b;border-radius:8px;
                        font-size:11px;color:#475569;line-height:1.7;">
              <b style="color:#64748b;">Підтримувані формати:</b> JPG, PNG, WEBP<br>
              <b style="color:#64748b;">Рекомендований розмір:</b> мін. 224×224 px<br>
              <b style="color:#64748b;">Поріг виявлення:</b> 50% ймовірності перелому
            </div>
            """)

        # ── Right column: results ─────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML('<div style="font-size:13px;color:#94a3b8;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px;">'
                    'Результат аналізу</div>')

            img_output = gr.Image(
                type='pil',
                label='',
                elem_id='result-img',
                show_label=False,
                height=220,
            )

            html_output = gr.HTML(
                value=_empty_html('Завантажте знімок і натисніть «Аналізувати»')
            )

    gr.HTML('<div id="footer">FracDetect AI · Університетська coursework · '
            'Не є медичним діагностичним інструментом</div>')

    btn.click(fn=analyze, inputs=img_input, outputs=[img_output, html_output])

if __name__ == '__main__':
    demo.launch(inbrowser=True)
