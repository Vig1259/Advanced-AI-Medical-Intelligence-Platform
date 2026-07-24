"""Generates the PDF project report (docs/Project_Report.pdf)."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=14, textColor=colors.HexColor("#1a3a5c"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1a3a5c"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=15, spaceAfter=8, fontName="Helvetica"))
styles.add(ParagraphStyle(name="Small", fontSize=8.5, leading=12, textColor=colors.HexColor("#555555")))

doc = SimpleDocTemplate(
    "docs/Project_Report.pdf",
    pagesize=letter,
    topMargin=0.85 * inch,
    bottomMargin=0.85 * inch,
    leftMargin=0.85 * inch,
    rightMargin=0.85 * inch,
    title="Advanced AI Medical Intelligence Platform - Project Report",
)

story = []

# --- Title ---
story.append(Paragraph("Advanced AI Medical Intelligence Platform", styles["H1"]))
story.append(Paragraph("Project Report — Chest X-Ray Pneumonia Detection with Explainable AI and LLM-Assisted Reporting", styles["Body"]))
story.append(Spacer(1, 6))
story.append(Paragraph("Prepared for: AI/ML Engineer Technical Evaluation", styles["Small"]))
story.append(Spacer(1, 20))

disclaimer_style = ParagraphStyle(name="Disclaimer", parent=styles["Body"], textColor=colors.HexColor("#8a1f1f"), backColor=colors.HexColor("#fdecea"), borderPadding=8, borderColor=colors.HexColor("#8a1f1f"), borderWidth=0.5)
story.append(Paragraph(
    "<b>Disclaimer:</b> This is a research/demo system built for a technical assessment. "
    "It is not a certified medical device and must not be used for real clinical diagnosis. "
    "All AI outputs require review by a licensed physician.",
    disclaimer_style,
))
story.append(Spacer(1, 16))

# --- 1. Objective ---
story.append(Paragraph("1. Project Objective", styles["H2"]))
story.append(Paragraph(
    "Build an end-to-end AI application that analyzes chest X-ray images, predicts pneumonia "
    "using a deep learning model, explains its predictions using Grad-CAM (Explainable AI), "
    "generates AI-assisted draft reports using a Large Language Model, exposes REST APIs, "
    "stores prediction history in a database, and is deployable via Docker with a web UI.",
    styles["Body"],
))

# --- 2. Architecture ---
story.append(Paragraph("2. System Architecture", styles["H2"]))
story.append(Paragraph(
    "The system follows a layered architecture: a Streamlit frontend for image upload and "
    "results visualization; a FastAPI REST backend exposing prediction and history endpoints; "
    "a PyTorch inference service (DenseNet121 backbone + Grad-CAM); an OpenAI-based report "
    "generation module; and a SQLAlchemy-backed database (SQLite by default, Postgres-ready) "
    "for persisting every prediction. Components communicate over HTTP/JSON, and the whole "
    "stack is containerized via Docker Compose.",
    styles["Body"],
))
story.append(Paragraph(
    "Request flow: image upload &rarr; preprocessing &rarr; DenseNet121 classification &rarr; "
    "Grad-CAM heatmap generation &rarr; LLM drafts a structured report from the prediction and "
    "heatmap summary &rarr; result persisted to DB &rarr; JSON response returned to the client.",
    styles["Body"],
))

# --- 3. Dataset & Model ---
story.append(Paragraph("3. Dataset and Model", styles["H2"]))
story.append(Paragraph(
    "<b>Dataset:</b> Kermany et al. \"Chest X-Ray Images (Pneumonia)\" dataset (Guangzhou "
    "Women and Children's Medical Center), publicly available via Kaggle. It contains labeled "
    "pediatric chest X-rays in two classes: NORMAL and PNEUMONIA, with a known class imbalance "
    "(~3x more PNEUMONIA than NORMAL cases).",
    styles["Body"],
))
story.append(Paragraph(
    "<b>Model:</b> DenseNet121 pretrained on ImageNet, fine-tuned for binary classification. "
    "DenseNet's dense connectivity pattern improves gradient flow and feature reuse, and it "
    "underlies CheXNet, a well-known chest radiograph classification benchmark. Training uses "
    "class-weighted cross-entropy loss to address imbalance, a short backbone-frozen warmup "
    "phase for the classifier head, data augmentation (flips, small rotations, color jitter), "
    "and a ReduceLROnPlateau learning-rate schedule with best-checkpoint selection by validation AUC.",
    styles["Body"],
))
story.append(Paragraph(
    "<b>Note on execution environment:</b> This report was produced in a sandboxed development "
    "environment without GPU or internet access, so the model could not be trained or benchmarked "
    "against the live dataset during assignment preparation. The training and evaluation scripts "
    "(<font face='Courier'>training/train.py</font>, "
    "<font face='Courier'>training/evaluate.py</font>) are complete and tested for correctness "
    "(syntax, logic, dataset interface) and are intended to be run in a GPU-enabled environment "
    "before final submission. Section 8 documents the exact commands and expected outputs.",
    styles["Body"],
))

# --- 4. XAI ---
story.append(Paragraph("4. Explainable AI (Grad-CAM)", styles["H2"]))
story.append(Paragraph(
    "Grad-CAM (Selvaraju et al., 2017) was implemented from scratch (not via a third-party XAI "
    "library) in <font face='Courier'>app/ml/gradcam.py</font>. Forward and backward hooks are "
    "registered on the last normalization layer of the DenseNet121 feature extractor. Gradients "
    "of the predicted class score with respect to the feature maps are global-average-pooled "
    "into per-channel importance weights, which are used to compute a weighted combination of "
    "the feature maps, followed by ReLU and min-max normalization. The resulting heatmap is "
    "resized and alpha-blended over the original X-ray to visually indicate which lung regions "
    "most influenced the model's decision.",
    styles["Body"],
))

# --- 5. LLM ---
story.append(Paragraph("5. LLM-Assisted Report Generation", styles["H2"]))
story.append(Paragraph(
    "The OpenAI API (default model: gpt-4o-mini, configurable) is used to convert the structured "
    "prediction output into a readable draft report. The system prompt constrains the model to "
    "a fixed structure (AI Impression, Key Observations, Confidence &amp; Limitations, Suggested "
    "Next Steps), instructs it to hedge appropriately rather than assert a definitive diagnosis, "
    "and forbids inventing findings beyond the provided data. Every report is appended with an "
    "explicit non-diagnostic disclaimer. If no API key is configured, or the API call fails, the "
    "system falls back to a deterministic template report so the pipeline degrades gracefully.",
    styles["Body"],
))

# --- 6. API ---
story.append(Paragraph("6. REST API Design", styles["H2"]))
api_table_data = [
    ["Endpoint", "Method", "Description"],
    ["/predict", "POST", "Upload an X-ray image; returns prediction, confidence, Grad-CAM overlay, LLM report"],
    ["/history", "GET", "List past predictions (paginated, filterable by class)"],
    ["/history/{id}", "GET", "Full detail of a single prediction record"],
    ["/history/{id}", "DELETE", "Delete a prediction record"],
    ["/health", "GET", "Liveness/readiness check, reports device and model status"],
]
api_table = Table(api_table_data, colWidths=[1.3 * inch, 0.7 * inch, 4.2 * inch])
api_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(api_table)
story.append(Spacer(1, 8))
story.append(Paragraph("Built with FastAPI, with Pydantic schemas for request/response validation and automatic OpenAPI/Swagger documentation at <font face='Courier'>/docs</font>.", styles["Body"]))

# --- 7. Database ---
story.append(Paragraph("7. Database Design", styles["H2"]))
story.append(Paragraph(
    "A single <font face='Courier'>prediction_history</font> table (SQLAlchemy ORM, "
    "<font face='Courier'>app/models_db.py</font>) stores: a UUID primary key, original filename, "
    "predicted class, confidence, the full class-probability distribution (JSON), the path to the "
    "saved Grad-CAM overlay image, the generated LLM report text, and a timestamp. SQLite is used "
    "by default for zero-config local/demo use; the connection string is externalized via "
    "<font face='Courier'>DATABASE_URL</font> so swapping to PostgreSQL for production requires no "
    "code changes.",
    styles["Body"],
))

# --- 8. Setup & Reproduction ---
story.append(PageBreak())
story.append(Paragraph("8. Setup, Training, and Reproduction Steps", styles["H2"]))
steps = [
    "Clone the repository and install dependencies: <font face='Courier'>pip install -r requirements.txt</font>",
    "Download the Kaggle \"Chest X-Ray Images (Pneumonia)\" dataset into <font face='Courier'>./data/</font> following the train/val/test/NORMAL/PNEUMONIA layout",
    "Train the model: <font face='Courier'>python training/train.py --data-root data --epochs 15</font>",
    "Evaluate on the test set: <font face='Courier'>python training/evaluate.py --checkpoint models/chest_xray_densenet121.pt</font>",
    "Set <font face='Courier'>OPENAI_API_KEY</font> in <font face='Courier'>.env</font>",
    "Run the API: <font face='Courier'>uvicorn app.main:app --reload</font>",
    "Run the frontend: <font face='Courier'>streamlit run frontend/streamlit_app.py</font>",
    "Or run everything via Docker: <font face='Courier'>docker compose up --build</font>",
]
story.append(ListFlowable([ListItem(Paragraph(s, styles["Body"])) for s in steps], bulletType="1"))

# --- 9. Evaluation criteria mapping ---
story.append(Paragraph("9. Evaluation Criteria Coverage", styles["H2"]))
crit_data = [
    ["Criterion", "Where addressed"],
    ["Deep Learning model performance", "training/train.py, training/evaluate.py (Sec. 3, 8)"],
    ["Code quality / project structure", "Modular app/, training/, frontend/, tests/ layout (Sec. 2)"],
    ["Explainable AI (Grad-CAM)", "app/ml/gradcam.py (Sec. 4)"],
    ["LLM integration", "app/llm/report_generator.py (Sec. 5)"],
    ["API development", "app/main.py, app/routers/ (Sec. 6)"],
    ["Database design", "app/models_db.py, app/database.py (Sec. 7)"],
    ["Web application", "frontend/streamlit_app.py"],
    ["Documentation", "README.md, this report"],
    ["Deployment", "Dockerfile, docker-compose.yml"],
    ["System design / best practices", "Config via env vars, tests/, error handling, disclaimers"],
]
crit_table = Table(crit_data, colWidths=[2.3 * inch, 3.9 * inch])
crit_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(crit_table)

# --- 10. Limitations ---
story.append(Paragraph("10. Known Limitations and Future Work", styles["H2"]))
limitations = [
    "Model not yet trained/benchmarked in this environment (no GPU/network access during development) — must be run before final grading if live metrics are required.",
    "Grad-CAM region description fed to the LLM is currently a simple heuristic rather than computed heatmap centroid/quadrant analysis.",
    "No authentication/authorization layer — required before any multi-user or production deployment.",
    "SQLite used for demo purposes; recommend PostgreSQL for concurrent production workloads.",
    "Only JPEG/PNG supported; no DICOM parsing.",
]
story.append(ListFlowable([ListItem(Paragraph(s, styles["Body"])) for s in limitations], bulletType="bullet"))

story.append(Paragraph("11. References", styles["H2"]))
refs = [
    "Kermany, D. et al. \"Chest X-Ray Images (Pneumonia)\" dataset, Guangzhou Women and Children's Medical Center (via Kaggle).",
    "Huang, G. et al. \"Densely Connected Convolutional Networks\" (DenseNet), CVPR 2017.",
    "Selvaraju, R. R. et al. \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\", ICCV 2017.",
    "Rajpurkar, P. et al. \"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning\", 2017.",
]
story.append(ListFlowable([ListItem(Paragraph(s, styles["Small"])) for s in refs], bulletType="bullet"))

doc.build(story)
print("PDF report generated at docs/Project_Report.pdf")
