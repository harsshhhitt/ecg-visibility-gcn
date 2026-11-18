"""Visual PDF report for a single-beat prediction.

    python src/report.py samples/demo_V.png --out report.pdf

Combines the prediction, class probabilities, MC-dropout uncertainty, saliency
explanation, visibility-graph statistics and plain-language advice into a
shareable one/two-page PDF. Advice is general educational information, not a
diagnosis.
"""

import argparse
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    HRFlowable, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from config import AAMI_CLASSES  # noqa: E402
from explain import node_saliency, plot_explanation  # noqa: E402
from predict import predict_beat, predict_with_uncertainty  # noqa: E402
from visibility_graph import graph_stats  # noqa: E402

NAVY = colors.HexColor("#14213d")
GREY = colors.HexColor("#5c6370")
LIGHT = colors.HexColor("#f2f4f8")
CLASS_COLOR = {"N": "#2a9d8f", "S": "#e9c46a", "V": "#e76f51", "F": "#d62828", "Q": "#6c757d"}

INFO = {
    "N": ("Normal beat", "The beat resembles a normal sinus beat.",
          ["A single normal-looking beat is not a health assessment; rhythm over time matters more.",
           "If you have symptoms (palpitations, dizziness, chest pain, fainting), seek medical advice regardless of this result.",
           "Keep routine check-ups; share any recorded ECG with your clinician."]),
    "S": ("Supraventricular ectopic beat", "An early beat that originates above the ventricles (atria / AV junction), e.g. a premature atrial contraction.",
          ["Isolated supraventricular beats are very common and usually harmless in healthy people.",
           "Frequent runs, or accompanying palpitations / breathlessness, are worth discussing with a clinician.",
           "Common triggers worth noting: caffeine, alcohol, stress, poor sleep, some stimulants and medications."]),
    "V": ("Ventricular ectopic beat", "An early, wide beat that originates in the ventricles, e.g. a premature ventricular contraction (PVC).",
          ["Occasional PVCs are common and often benign, but frequent PVCs deserve clinical review.",
           "Seek prompt medical attention if accompanied by chest pain, fainting, severe dizziness or shortness of breath.",
           "Bring a longer ECG / Holter recording to your clinician; a single beat cannot show burden or pattern."]),
    "F": ("Fusion beat", "A beat formed when a normal and a ventricular impulse activate the heart at the same moment.",
          ["Fusion beats usually appear alongside ventricular ectopy and are interpreted in that context.",
           "This is the class the model is least reliable on (see model limitations) - treat as a flag to verify, not a finding.",
           "Ask a clinician to review the full recording."]),
    "Q": ("Unclassifiable beat", "Paced, fused-paced or otherwise unclassifiable beat.",
          ["The model has seen very few examples of this class; the result is largely unreliable.",
           "Retake a cleaner, well-cropped image of a single beat or use the original ECG recording.",
           "Have a clinician review the source ECG."]),
}


def _png(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _prob_chart(probs, label):
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    names = AAMI_CLASSES[::-1]
    vals = [probs[c] for c in names]
    bars = ax.barh(names, vals, color=[CLASS_COLOR[c] if c == label else "#cfd4dc" for c in names])
    for b, v in zip(bars, vals):
        ax.text(min(v + 0.015, 0.93), b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return _png(fig)


def _gauge(entropy):
    fig, ax = plt.subplots(figsize=(6.2, 0.9))
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=[0, 1, 0, 1], aspect="auto", cmap="RdYlGn_r")
    ax.axvline(entropy, color="black", lw=3)
    ax.plot([entropy], [1.18], marker="v", color="black", clip_on=False, markersize=9)
    ax.set_yticks([])
    ax.set_xticks([0, 0.3, 0.6, 1])
    ax.set_xticklabels(["certain", "", "", "very uncertain"], fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return _png(fig)


def _explain_fig(waveform, label, probs):
    sal, _, _ = node_saliency(waveform)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    plot_explanation(waveform, sal, label, probs, ax_list=axes)
    fig.tight_layout()
    return _png(fig)


def _uncertainty_band(entropy):
    if entropy < 0.35:
        return "Low", "#2a9d8f", "The model's repeated passes agree closely."
    if entropy < 0.6:
        return "Moderate", "#e9a23b", "The model is somewhat unsure; read the result as indicative only."
    return "High", "#d62828", "The model's passes disagree or spread across classes; do not rely on this result."


def build_report(waveform, out_path, source_name="uploaded image", source_image=None, mc_samples=30):
    label, probs, data, _ = predict_beat(waveform)
    mean_p, std_p, entropy = predict_with_uncertainty(waveform, n_samples=mc_samples)
    stats = graph_stats(data)
    title, blurb, advice = INFO[label]
    band, band_color, band_text = _uncertainty_band(entropy)
    conf = probs[label]

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=NAVY, fontSize=22, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13.5)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=GREY, leading=11)
    banner_txt = ParagraphStyle("bt", parent=body, textColor=colors.white, fontSize=11, leading=15)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title="ECG Beat Analysis Report",
                            author="ECG Visibility-Graph GCN")
    W = A4[0] - 32 * mm
    story = [
        Paragraph("ECG Beat Analysis Report", h1),
        Paragraph(f"Visibility-graph GCN &middot; generated {datetime.now():%d %b %Y, %H:%M} &middot; source: {source_name}", small),
        HRFlowable(width="100%", color=NAVY, thickness=1.2, spaceBefore=4, spaceAfter=8),
    ]

    # verdict banner
    big = ParagraphStyle("big", parent=banner_txt, fontSize=20, leading=26)
    verdict = Table([[Paragraph("PREDICTED CLASS", ParagraphStyle("k", parent=banner_txt, fontSize=8, leading=11))],
                     [Paragraph(f"<b>{label} &mdash; {title}</b>", big)],
                     [Paragraph(f"Confidence {conf:.1%} &nbsp;|&nbsp; Uncertainty: <b>{band}</b>", banner_txt)]],
                    colWidths=[W])
    verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CLASS_COLOR[label])),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story += [verdict, Spacer(1, 6), Paragraph(blurb, body)]

    # input image + probabilities side by side
    left = []
    if source_image is not None:
        left = [Image(source_image, width=W * 0.36, height=W * 0.28, kind="proportional")]
    right = [Image(_prob_chart(probs, label), width=W * 0.6, height=W * 0.24, kind="proportional")]
    story += [Paragraph("Class probabilities", h2),
              Table([[left or "", right]], colWidths=[W * 0.38, W * 0.62],
                    style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")])]

    # uncertainty
    story += [Paragraph("Reliability of this prediction", h2),
              Image(_gauge(entropy), width=W, height=W * 0.15, kind="proportional"),
              Paragraph(f"<b>Predictive entropy {entropy:.2f}</b> (0 = certain, 1 = maximally uncertain) - "
                        f"<font color='{band_color}'><b>{band}</b></font>. {band_text} "
                        f"MC-dropout probability for {label}: {mean_p[label]:.1%} &plusmn; {std_p[label]:.1%}.", body)]

    # explanation
    story.append(KeepTogether([
        Paragraph("What the model looked at", h2),
        Image(_explain_fig(waveform, label, probs), width=W, height=W * 0.35, kind="proportional"),
        Paragraph("Bright points are the samples of the beat that contributed most to the predicted class "
                  "(gradient x input saliency). Node size in the graph shows the same importance.", small)]))

    # graph stats table
    rows = [["Nodes", "Edges", "Density", "Mean degree", "Max degree", "Hub position"],
            [stats["nodes"], stats["edges"], f"{stats['density']:.3f}", f"{stats['mean_degree']:.1f}",
             stats["max_degree"], f"{stats['hub_position']:.0%} along beat"]]
    t = Table(rows, colWidths=[W / 6] * 6)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), LIGHT), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([Paragraph("Visibility-graph structure", h2), t]))

    # advice
    bullets = list(advice)
    if band == "High":
        bullets.insert(0, "Because uncertainty is high, retake a clearer, tightly cropped image of one beat and re-run before drawing any conclusion.")
    if conf < 0.6:
        bullets.insert(0, f"Confidence is only {conf:.0%}; the runner-up class was "
                          f"{sorted(probs, key=probs.get)[-2]} ({sorted(probs.values())[-2]:.0%}).")
    adv = Table([[Paragraph("&bull; " + b, body)] for b in bullets], colWidths=[W])
    adv.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(KeepTogether([Paragraph("Advice &amp; suggestions", h2), adv]))

    tips = ["Crop tightly around one P-QRS-T complex; remove text, labels and grid where possible.",
            "Use a dark trace on a light background and adjust the darkness threshold if the trace is faint.",
            "For real interpretation use a longer (10 s+) recording - beat classification cannot assess rhythm.",
            "Compare with a second model/backbone (--arch gat) before trusting a rare-class prediction."]
    story.append(KeepTogether([Paragraph("Getting a better result", h2)] +
                              [Paragraph("&bull; " + x, body) for x in tips]))

    story += [Paragraph("Model limitations", h2),
              Paragraph("Calibrated 3-model ensemble (GCN + GraphSAGE + GATv2) trained on MIT-BIH with an inter-patient "
                        "DS1/DS2 split. On held-out DS2 patients: accuracy ~0.89, macro-AUROC ~0.81, F1 ~0.94 for N and "
                        "~0.62 for V. Supraventricular (S), fusion (F) and unclassifiable (Q) beats are <b>not reliably "
                        "detected</b> from a single beat (recall ~0 at the default setting), so a prediction of N does "
                        "not rule them out. Images are digitized heuristically without a mV calibration.", body),
              Spacer(1, 8),
              HRFlowable(width="100%", color=GREY, thickness=0.5),
              Paragraph("<b>Disclaimer:</b> This report is generated by a research/demo model and is <b>not a medical "
                        "diagnosis</b>. It must not be used for clinical decisions. Consult a qualified healthcare "
                        "professional for interpretation of any ECG.", small)]

    doc.build(story)
    return label, probs, entropy


if __name__ == "__main__":
    from image_digitize import digitize_waveform

    parser = argparse.ArgumentParser(description="Generate a PDF report for an ECG beat image.")
    parser.add_argument("image")
    parser.add_argument("--out", default="ecg_report.pdf")
    parser.add_argument("--threshold", type=int, default=130)
    args = parser.parse_args()
    wf = digitize_waveform(args.image, threshold=args.threshold)
    lab, _, ent = build_report(wf, args.out, source_name=args.image.replace("\\", "/").split("/")[-1],
                               source_image=args.image)
    print(f"{lab} (uncertainty {ent:.2f}) -> {args.out}")
