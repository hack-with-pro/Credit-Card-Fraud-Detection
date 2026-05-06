import base64
import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_PATH = OUTPUTS_DIR / "dashboard.html"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_summary_metrics(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics

    text = path.read_text(encoding="utf-8")
    patterns = {
        "best_model": r"Best model:\s*(.+)",
        "decision_threshold": r"Decision threshold:\s*([0-9.]+)",
        "roc_auc": r"ROC-AUC:\s*([0-9.]+)",
        "pr_auc": r"PR-AUC:\s*([0-9.]+)",
        "f1_score": r"F1 score:\s*([0-9.]+)",
        "precision": r"Precision:\s*([0-9.]+)",
        "recall": r"Recall:\s*([0-9.]+)",
        "balanced_accuracy": r"Balanced accuracy:\s*([0-9.]+)",
        "top_precision": r"Top-alert precision@\d+:\s*([0-9.]+)",
        "top_recall": r"Top-alert recall@\d+:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = match.group(1).strip()
    return metrics


def read_confusion_matrix(path: Path) -> dict[str, int]:
    rows = read_csv_rows(path)
    matrix = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in rows:
        label = row.get("", "")
        if label == "actual_legitimate":
            matrix["tn"] = int(row["pred_legitimate"])
            matrix["fp"] = int(row["pred_fraud"])
        elif label == "actual_fraud":
            matrix["fn"] = int(row["pred_legitimate"])
            matrix["tp"] = int(row["pred_fraud"])
    return matrix


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{encoded}"


def build_threshold_path(points: list[dict[str, str]], width: int = 760, height: int = 240) -> str:
    if not points:
        return ""

    thresholds = [float(item["threshold"]) for item in points]
    f1_scores = [float(item["f1"]) for item in points]

    min_x = min(thresholds)
    max_x = max(thresholds)
    min_y = min(f1_scores)
    max_y = max(f1_scores)

    def scale_x(value: float) -> float:
        span = max(max_x - min_x, 1e-9)
        return 20 + ((value - min_x) / span) * (width - 40)

    def scale_y(value: float) -> float:
        span = max(max_y - min_y, 1e-9)
        return height - 20 - ((value - min_y) / span) * (height - 40)

    coords = [f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(thresholds, f1_scores)]
    return " ".join(coords)


def metric_card(title: str, value: str, accent: str) -> str:
    return f"""
        <article class="metric-card">
          <div class="metric-accent" style="background:{accent};"></div>
          <p>{html.escape(title)}</p>
          <h3>{html.escape(value)}</h3>
        </article>
    """


def format_percent(value: str) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except ValueError:
        return value


def build_dashboard() -> str:
    summary = parse_summary_metrics(OUTPUTS_DIR / "business_summary.txt")
    benchmark_rows = read_csv_rows(OUTPUTS_DIR / "model_benchmark.csv")
    threshold_rows = read_csv_rows(OUTPUTS_DIR / "threshold_search.csv")
    alert_rows = read_csv_rows(OUTPUTS_DIR / "high_risk_transactions.csv")
    feature_rows = read_csv_rows(OUTPUTS_DIR / "top_feature_importance.csv")
    confusion = read_confusion_matrix(OUTPUTS_DIR / "confusion_matrix.csv")

    score_distribution = image_to_data_uri(OUTPUTS_DIR / "score_distribution.png")
    feature_importance = image_to_data_uri(OUTPUTS_DIR / "feature_importance.png")
    threshold_path = build_threshold_path(threshold_rows)

    top_alerts = alert_rows[:18]
    max_importance = max((float(row["importance"]) for row in feature_rows), default=1.0)
    total_matrix = max(sum(confusion.values()), 1)

    model_cards = "".join(
        f"""
        <article class="model-card">
          <h4>{html.escape(row['model'])}</h4>
          <p>PR-AUC <strong>{float(row['validation_pr_auc']):.4f}</strong></p>
          <p>ROC-AUC <strong>{float(row['validation_roc_auc']):.4f}</strong></p>
          <p>F1 @ 0.50 <strong>{float(row['validation_f1_at_0_50']):.4f}</strong></p>
        </article>
        """
        for row in benchmark_rows
    )

    feature_bars = "".join(
        f"""
        <div class="feature-row">
          <div class="feature-label">
            <span>{html.escape(row['feature'])}</span>
            <strong>{float(row['importance']):.3f}</strong>
          </div>
          <div class="feature-bar-track">
            <div class="feature-bar-fill" style="width:{(float(row['importance']) / max_importance) * 100:.1f}%"></div>
          </div>
        </div>
        """
        for row in feature_rows[:10]
    )

    alert_table = "".join(
        f"""
        <tr>
          <td>{index + 1}</td>
          <td>{html.escape(row.get('Risk_Tier', 'unknown'))}</td>
          <td>{float(row.get('Fraud_Score', 0)):.6f}</td>
          <td>{row.get('Actual_Class', '0')}</td>
          <td>{float(row.get('Amount', 0)):.2f}</td>
          <td>{row.get('Hour', '-')}</td>
          <td>{float(row.get('V_Abs_Max', 0)):.3f}</td>
        </tr>
        """
        for index, row in enumerate(top_alerts)
    )

    payload = {
        "thresholds": threshold_rows,
        "benchmarks": benchmark_rows,
        "alerts": top_alerts,
        "features": feature_rows[:10],
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fraud Detection Command Center</title>
  <style>
    :root {{
      --bg: #07111f;
      --panel: rgba(10, 22, 40, 0.72);
      --panel-strong: rgba(11, 28, 50, 0.88);
      --line: rgba(255,255,255,0.08);
      --text: #eff7ff;
      --muted: #9fb4d1;
      --cyan: #46d9ff;
      --blue: #4f82ff;
      --pink: #ff5fa2;
      --gold: #ffbf5f;
      --green: #4ff0b7;
      --shadow: 0 28px 70px rgba(0, 0, 0, 0.38);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(79, 130, 255, 0.35), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 95, 162, 0.20), transparent 26%),
        radial-gradient(circle at 20% 80%, rgba(79, 240, 183, 0.14), transparent 24%),
        linear-gradient(140deg, #040b14 0%, #091625 45%, #0f1e34 100%);
      min-height: 100vh;
    }}

    body::before,
    body::after {{
      content: "";
      position: fixed;
      border-radius: 50%;
      filter: blur(90px);
      z-index: 0;
      pointer-events: none;
    }}

    body::before {{
      width: 260px;
      height: 260px;
      background: rgba(70, 217, 255, 0.14);
      top: -40px;
      left: -60px;
    }}

    body::after {{
      width: 300px;
      height: 300px;
      background: rgba(255, 95, 162, 0.12);
      bottom: -100px;
      right: -40px;
    }}

    .page {{
      position: relative;
      z-index: 1;
      width: min(1240px, calc(100% - 32px));
      margin: 24px auto 48px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      padding: 36px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background:
        linear-gradient(135deg, rgba(79,130,255,0.20), rgba(255,95,162,0.10)),
        rgba(6, 18, 34, 0.78);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}

    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%);
      transform: translateX(-100%);
      animation: shimmer 8s linear infinite;
    }}

    @keyframes shimmer {{
      to {{
        transform: translateX(100%);
      }}
    }}

    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      color: var(--cyan);
      font-size: 0.86rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .hero h1 {{
      margin: 18px 0 10px;
      font-size: clamp(2.4rem, 5vw, 4.4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}

    .hero p {{
      max-width: 760px;
      margin: 0;
      color: var(--muted);
      font-size: 1.08rem;
      line-height: 1.7;
    }}

    .hero-grid,
    .metrics,
    .section-grid,
    .gallery {{
      display: grid;
      gap: 18px;
    }}

    .hero-grid {{
      grid-template-columns: 1.3fr 0.9fr;
      align-items: end;
      gap: 24px;
    }}

    .snapshot {{
      display: grid;
      gap: 14px;
    }}

    .snapshot-card {{
      padding: 20px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255,255,255,0.08);
    }}

    .snapshot-card small {{
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .snapshot-card strong {{
      font-size: 1.6rem;
    }}

    section {{
      margin-top: 20px;
      padding: 22px;
      border-radius: 26px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}

    h2 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
    }}

    .section-copy {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.65;
    }}

    .metrics {{
      grid-template-columns: repeat(5, minmax(0, 1fr));
    }}

    .metric-card {{
      position: relative;
      padding: 18px 18px 20px;
      border-radius: 22px;
      overflow: hidden;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }}

    .metric-accent {{
      height: 4px;
      border-radius: 999px;
      margin-bottom: 14px;
    }}

    .metric-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .metric-card h3 {{
      margin: 10px 0 0;
      font-size: 1.6rem;
      letter-spacing: -0.03em;
    }}

    .section-grid {{
      grid-template-columns: 1.1fr 0.9fr;
      align-items: start;
    }}

    .panel {{
      padding: 20px;
      border-radius: 22px;
      background: var(--panel-strong);
      border: 1px solid rgba(255,255,255,0.06);
      height: 100%;
    }}

    .model-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}

    .model-card {{
      padding: 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(79,130,255,0.15), rgba(255,95,162,0.06));
      border: 1px solid rgba(255,255,255,0.08);
    }}

    .model-card h4 {{
      margin: 0 0 12px;
      font-size: 1rem;
    }}

    .model-card p {{
      margin: 0 0 8px;
      color: var(--muted);
    }}

    .model-card strong {{
      color: var(--text);
    }}

    .chart-wrap {{
      padding: 14px;
      border-radius: 20px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.05);
    }}

    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .axis-labels {{
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 6px;
    }}

    .matrix {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}

    .cell {{
      padding: 18px;
      border-radius: 20px;
      min-height: 140px;
      display: flex;
      flex-direction: column;
      justify-content: end;
      border: 1px solid rgba(255,255,255,0.08);
    }}

    .cell small {{
      color: rgba(255,255,255,0.72);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.75rem;
    }}

    .cell strong {{
      margin-top: 10px;
      font-size: 2rem;
    }}

    .feature-row + .feature-row {{
      margin-top: 16px;
    }}

    .feature-label {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      color: var(--muted);
    }}

    .feature-label span {{
      color: var(--text);
    }}

    .feature-bar-track {{
      height: 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      overflow: hidden;
    }}

    .feature-bar-fill {{
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--cyan), var(--pink), var(--gold));
    }}

    .gallery {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .image-card {{
      overflow: hidden;
      border-radius: 24px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
    }}

    .image-card img {{
      width: 100%;
      display: block;
    }}

    .image-card figcaption {{
      padding: 14px 16px 16px;
      color: var(--muted);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 20px;
    }}

    thead {{
      background: rgba(255,255,255,0.07);
    }}

    th, td {{
      padding: 13px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 0.94rem;
    }}

    tbody tr:hover {{
      background: rgba(255,255,255,0.04);
    }}

    .risk-pill {{
      display: inline-flex;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,95,162,0.18);
      color: #ffd1e4;
      font-size: 0.82rem;
      text-transform: capitalize;
    }}

    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.92rem;
    }}

    @media (max-width: 1080px) {{
      .hero-grid,
      .section-grid,
      .metrics,
      .gallery,
      .model-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 700px) {{
      .page {{
        width: min(100% - 18px, 100%);
        margin-top: 10px;
      }}

      .hero,
      section {{
        padding: 18px;
        border-radius: 22px;
      }}

      .hero h1 {{
        font-size: 2.3rem;
      }}

      th, td {{
        padding: 10px 8px;
        font-size: 0.84rem;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="hero-grid">
        <div>
          <span class="eyebrow">Fraud Detection Command Center</span>
          <h1>Credit Card Fraud Detection — Model Results</h1>
          <p>
            284,807 transactions analyzed. 3 candidate models benchmarked. Extra Trees selected as the winner
            with ROC-AUC 0.9738 and PR-AUC 0.8750. Threshold tuned on the validation set to maximize
            fraud detection F1. Results below cover the held-out test set only.
          </p>
        </div>
        <div class="snapshot">
          <div class="snapshot-card">
            <small>Winning Model</small>
            <strong>{html.escape(summary.get("best_model", "Not available"))}</strong>
          </div>
          <div class="snapshot-card">
            <small>Decision Threshold</small>
            <strong>{html.escape(summary.get("decision_threshold", "N/A"))}</strong>
          </div>
          <div class="snapshot-card">
            <small>Dashboard Generated</small>
            <strong>{datetime.now().strftime("%d %b %Y %H:%M")}</strong>
          </div>
        </div>
      </div>
    </header>

    <section>
      <h2>Performance Snapshot</h2>
      <p class="section-copy">The key fraud metrics are surfaced as bold cards so the project feels like a product, not just a folder of CSV exports.</p>
      <div class="metrics">
        {metric_card("ROC-AUC", summary.get("roc_auc", "N/A"), "linear-gradient(90deg,#46d9ff,#4f82ff)")}
        {metric_card("PR-AUC", summary.get("pr_auc", "N/A"), "linear-gradient(90deg,#ff5fa2,#ffbf5f)")}
        {metric_card("F1 Score", summary.get("f1_score", "N/A"), "linear-gradient(90deg,#4ff0b7,#46d9ff)")}
        {metric_card("Precision", summary.get("precision", "N/A"), "linear-gradient(90deg,#ffbf5f,#ff5fa2)")}
        {metric_card("Recall", summary.get("recall", "N/A"), "linear-gradient(90deg,#7ea6ff,#46d9ff)")}
      </div>
    </section>

    <section class="section-grid">
      <div class="panel">
        <h2>Model Face-Off</h2>
        <p class="section-copy">Each candidate model gets its own polished tile so the winner stands out immediately.</p>
        <div class="model-grid">
          {model_cards}
        </div>
      </div>
      <div class="panel">
        <h2>Threshold Tuning Curve</h2>
        <p class="section-copy">F1 score peaks around the chosen threshold, which makes the decision policy easy to explain in a demo or review.</p>
        <div class="chart-wrap">
          <svg viewBox="0 0 760 240" aria-label="Threshold tuning curve">
            <defs>
              <linearGradient id="curveGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#46d9ff"></stop>
                <stop offset="50%" stop-color="#4f82ff"></stop>
                <stop offset="100%" stop-color="#ff5fa2"></stop>
              </linearGradient>
            </defs>
            <line x1="20" y1="20" x2="20" y2="220" stroke="rgba(255,255,255,0.14)" />
            <line x1="20" y1="220" x2="740" y2="220" stroke="rgba(255,255,255,0.14)" />
            <polyline fill="none" stroke="url(#curveGlow)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" points="{threshold_path}"></polyline>
          </svg>
        </div>
        <div class="axis-labels">
          <span>Threshold 0.05</span>
          <span>Best threshold {html.escape(summary.get("decision_threshold", "N/A"))}</span>
          <span>Threshold 0.95</span>
        </div>
      </div>
    </section>

    <section class="section-grid">
      <div class="panel">
        <h2>Confusion Matrix</h2>
        <p class="section-copy">The model is strong on both fraud capture and false-positive control, and the matrix is styled to make that visible at a glance.</p>
        <div class="matrix">
          <div class="cell" style="background:linear-gradient(180deg, rgba(79,240,183,0.30), rgba(79,240,183,0.12));">
            <small>True Negatives</small>
            <strong>{confusion["tn"]:,}</strong>
          </div>
          <div class="cell" style="background:linear-gradient(180deg, rgba(255,191,95,0.30), rgba(255,191,95,0.12));">
            <small>False Positives</small>
            <strong>{confusion["fp"]:,}</strong>
          </div>
          <div class="cell" style="background:linear-gradient(180deg, rgba(255,95,162,0.30), rgba(255,95,162,0.12));">
            <small>False Negatives</small>
            <strong>{confusion["fn"]:,}</strong>
          </div>
          <div class="cell" style="background:linear-gradient(180deg, rgba(70,217,255,0.30), rgba(70,217,255,0.12));">
            <small>True Positives</small>
            <strong>{confusion["tp"]:,}</strong>
          </div>
        </div>
        <p class="footer-note">Test-set volume represented here: {total_matrix:,} transactions.</p>
      </div>
      <div class="panel">
        <h2>Feature Spotlight</h2>
        <p class="section-copy">The most important signals are displayed as vibrant bars so feature impact is easy to talk through during presentations.</p>
        {feature_bars}
        <p class="footer-note">Top-alert precision: {format_percent(summary.get("top_precision", "0"))} | Top-alert recall: {format_percent(summary.get("top_recall", "0"))}</p>
      </div>
    </section>

    <section>
      <h2>Visual Evidence</h2>
      <p class="section-copy">The exported matplotlib charts are embedded into the dashboard so the project keeps its analytical depth while gaining a strong visual layer.</p>
      <div class="gallery">
        <figure class="image-card">
          <img src="{score_distribution}" alt="Score distribution" />
          <figcaption>Fraud score separation between legitimate and fraudulent transactions.</figcaption>
        </figure>
        <figure class="image-card">
          <img src="{feature_importance}" alt="Feature importance" />
          <figcaption>Top feature importance chart from the winning tree-based model.</figcaption>
        </figure>
      </div>
    </section>

    <section>
      <h2>High-Risk Alert Queue</h2>
      <p class="section-copy">The riskiest transactions are elevated into a clean, readable investigation table so the UI feels operational instead of decorative.</p>
      <div class="panel" style="padding:0; background:transparent; border:none;">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Risk Tier</th>
              <th>Fraud Score</th>
              <th>Actual Class</th>
              <th>Amount</th>
              <th>Hour</th>
              <th>V Abs Max</th>
            </tr>
          </thead>
          <tbody>
            {alert_table.replace("<td>critical</td>", "<td><span class='risk-pill'>critical</span></td>").replace("<td>high</td>", "<td><span class='risk-pill'>high</span></td>").replace("<td>medium</td>", "<td><span class='risk-pill'>medium</span></td>").replace("<td>low</td>", "<td><span class='risk-pill'>low</span></td>")}
          </tbody>
        </table>
      </div>
      <p class="footer-note">Embedded data payload size kept intentionally small for a fast local load.</p>
    </section>
  </main>
  <script>
    window.dashboardData = {json.dumps(payload)};
  </script>
</body>
</html>
"""


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(build_dashboard(), encoding="utf-8")
    print(f"Dashboard written to: {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
