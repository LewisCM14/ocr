# OCR Pipeline

A production-grade, resumable batch OCR pipeline for processing large volumes of scanned TIFF images on Windows, with output designed for downstream Elasticsearch indexing.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Design](#pipeline-design)
  - [Discovery](#1-discovery)
  - [Database as Queue](#2-database-as-queue)
  - [Image Preprocessing](#3-image-preprocessing)
  - [OCR Engine](#4-ocr-engine)
  - [Output Format](#5-output-format)
  - [Multiprocessing](#6-multiprocessing)
  - [Resumability and Fault Tolerance](#7-resumability-and-fault-tolerance)
- [Design Decisions and Rationale](#design-decisions-and-rationale)
- [Getting Started](#getting-started)
- [Configuration Reference](#configuration-reference)
- [CLI Reference](#cli-reference)
- [Future Improvements](#future-improvements)

---

## Overview

The pipeline takes scanned TIFF images and converts the visual text into machine-encoded, searchable text. Outputs are written to the filesystem in plain text and structured JSON formats ready for ingestion into Elasticsearch.

The pipeline is:
- **Resumable** — crashes, reboots, and network blips do not lose progress
- **Parallel** — N worker processes run concurrently, each claiming work atomically from the database
- **Non-destructive** — the original TIFF files are never modified; outputs mirror the input directory tree
- **CPU-only today, GPU-ready tomorrow** — the OCR engine is abstracted behind an interface that can be swapped in config

---

## Project Structure

```
ocr/
├── environment.yml          # Conda environment (all deps from conda-forge)
├── config.yaml              # All tuneable settings — edit before running
├── setup_db.sql             # MSSQL schema: run once in SSMS or sqlcmd
│
├── pipeline/                # Core library
│   ├── config.py            # Typed dataclass config loader
│   ├── db.py                # SQLAlchemy 2.0 ORM models + engine factory
│   ├── discovery.py         # Filesystem walker and image registration
│   ├── preprocessor.py      # Image enhancement chain (deskew, denoise, binarise)
│   ├── ocr_engine.py        # OCR engine abstraction + Tesseract, EasyOCR, PaddleOCR
│   ├── worker.py            # Per-process work loop (claim → preprocess → OCR → write)
│   └── manager.py           # ProcessPoolExecutor orchestrator + progress monitor
│
└── scripts/                 # CLI entry points
    ├── discover.py          # Scan filesystem and register images in DB
    ├── process.py           # Launch the processing pipeline
    └── status.py            # Monitor progress, list failures, reset failed items
```

---

## Pipeline Design

### 1. Discovery

Before any OCR runs, `scripts/discover.py` walks the entire input directory tree and registers every TIFF file in the `ocr_images` database table. This separates the concerns of *finding* work from *doing* work, giving you an accurate total count up front and making it possible to run discovery again safely at any time as new images arrive.

Each row starts in `pending` status and is only updated once a worker claims it. Files already in the database are skipped, so re-running discovery against a partially processed dataset is safe.

### 2. Database as Queue

MSSQL is used as a distributed work queue. This is a deliberate choice over message brokers (RabbitMQ, Redis) because:

- The queue state is durable by default (transactional, survives reboots)
- Progress queries are trivial SQL — no additional monitoring tooling required
- The full audit trail (who processed what, when, how long it took, any errors) lives in the same place as the queue itself

Worker processes claim batches using a single atomic `UPDATE TOP(N) … OUTPUT INSERTED.*` statement. This is a well-established SQL Server pattern that is race-condition-free without needing advisory locks or application-level coordination: all N workers can hit the database simultaneously and each will receive a distinct, non-overlapping set of images.

```sql
UPDATE TOP (10) ocr_images
SET status = 'processing', worker_id = :worker_id, started_at = GETUTCDATE()
OUTPUT INSERTED.id, INSERTED.file_path, INSERTED.file_name
WHERE status = 'pending' AND retry_count < :max_retries
```

### 3. Image Preprocessing

Historical scans require substantial preparation before OCR. The preprocessing chain runs in order:

| Step | Purpose | Why it matters for old documents |
|---|---|---|
| **DPI normalisation** | Upscale images below `min_dpi` to `target_dpi` (default 300) using Lanczos resampling | Tesseract is calibrated for 300 DPI; lower-resolution inputs degrade accuracy significantly |
| **Grayscale conversion** | Reduce to single channel | Eliminates colour cast from paper ageing and yellowing |
| **Non-local means denoising** | Reduce scan noise while preserving edge detail | Old paper has texture noise and grain that confuses character recognition; NLM is gentler than Gaussian blur |
| **Deskewing** | Detect and correct page rotation | Hand-placed documents on a scanner flatbed are rarely perfectly straight; even 1–2° of tilt measurably reduces word accuracy |
| **Sauvola binarisation** | Adaptive local thresholding | The critical step for aged documents: handles uneven illumination, foxing, yellowing, and ink bleed-through that global Otsu thresholding cannot |

All steps are individually toggleable in `config.yaml`. For documents known to be high quality, steps can be disabled to increase throughput.

**Why Sauvola over Otsu?**
Otsu binarisation finds a single global threshold that divides the histogram into foreground and background. This works well for evenly lit, uniform paper. Old scans frequently have lighting gradients from document curvature, darkened margins, or uneven illumination from the scanner. Sauvola computes a local threshold for each pixel neighbourhood based on the local mean and standard deviation, making it tolerant of all of these effects at the cost of slightly higher computation.

### 4. OCR Engine

The OCR engine is decoupled behind an abstract `OCREngine` interface in `pipeline/ocr_engine.py`. The concrete engine is selected by a single config setting (`ocr.engine`). Three implementations are provided:

| Engine | Status | Licence | GPU support |
|---|---|---|---|
| **Tesseract 5** (LSTM) | Default, production-ready | Apache 2.0 | No |
| **EasyOCR** | Implemented, install separately | Apache 2.0 | Yes — `gpu=True` |
| **PaddleOCR** | Implemented, install separately | Apache 2.0 | Yes — `use_gpu=True` |

**Why Tesseract 5 as the default?**
- Ships on conda-forge — no additional installation steps
- The LSTM-based engine (`--oem 1`) handles a wider variety of typefaces, including older serif fonts, than the classic pattern-matching engine
- Well-understood, widely deployed, and easy to tune with config strings
- No GPU required

`--psm 3` (fully automatic page segmentation) is the default and handles mixed column layouts. For documents known to be single-column (e.g. typed letters, government forms), `--psm 6` will be more accurate and faster.

### 5. Output Format

For each TIFF file, up to three output files are written in a directory structure that mirrors the input tree. Which formats are produced is controlled by the `output.formats` list in `config.yaml`.

**`.txt`** — UTF-8 plain text, one file per source TIFF. Multi-page TIFFs have a `--- Page Break ---` separator between pages. Simple and human-readable.

**`.pdf`** — Searchable PDF. The preprocessed image is embedded as the visual layer with an invisible Unicode text layer precisely positioned over each recognised word by Tesseract. The file opens in any PDF reader (Adobe Acrobat, Edge, Sumatra, etc.) and supports Ctrl+F full-text search. Multi-page TIFFs produce a single multi-page PDF. Requires `ocr.engine = "tesseract"`.

**`.json`** — Structured document designed for direct Elasticsearch ingestion. Top-level fields include:

```json
{
  "source_path": "C:\\images\\1930\\batch001\\doc_00042.tif",
  "file_name": "doc_00042.tif",
  "relative_path": "1930/batch001/doc_00042",
  "page_count": 3,
  "full_text": "...",
  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "confidence": 0.87,
      "word_count": 412,
      "char_count": 2341,
      "processing_time_ms": 1840
    }
  ],
  "ocr_engine": "tesseract",
  "processed_at": "2026-06-19T14:23:01+00:00"
}
```

The `relative_path` and `source_path` fields allow the Elasticsearch documents to be linked back to both the original TIFF and the output files. `confidence` is a 0.0–1.0 score averaged from per-word Tesseract confidence values and can be used in the search application to surface low-confidence results for manual review.

**Enabling searchable PDF output** — uncomment `pdf` in `config.yaml`:

```yaml
output:
  formats:
    - "txt"
    - "json"
    - "pdf"   # add this line
```

PDF generation reuses the preprocessed image already in memory during the OCR pass — there is no second preprocessing or second Tesseract invocation. The per-page cost is small (Tesseract renders the PDF while producing the text). File sizes are larger than the `.txt` output but comparable to the original TIFF since the image is embedded.

### 6. Multiprocessing

On a large image dataset, single-process OCR would take weeks. The pipeline uses Python's `ProcessPoolExecutor` rather than threads because:

- CPython's GIL prevents true CPU parallelism with threads
- Each worker process imports the OCR engine independently, so there is no shared native library state
- The `spawn` start method (Windows default) means each worker is a clean process with no risk of inherited file handles or corrupted shared state

Each worker runs an independent loop: claim batch → process each image → update DB → repeat until no pending images remain. Workers are entirely self-contained; the manager process only monitors progress and waits for workers to finish.

**Sizing:** A reasonable starting point is `num_workers = CPU cores - 1`, leaving one core for the OS and database. OCR is CPU-bound and memory-moderate (each worker holds one image in memory at a time), so linear scaling with core count is achievable.

### 7. Resumability and Fault Tolerance

Every image has a `status` column (`pending`, `processing`, `complete`, `failed`) and a `retry_count`. The pipeline handles failures at multiple levels:

- **Worker crash mid-batch:** On next startup, any image stuck in `processing` for longer than `stale_processing_minutes` (default 60) is automatically reset to `pending`. The `retry_count` is incremented.
- **Repeated failures:** An image that fails `max_retries` times (default 3) is marked `failed` and excluded from future processing runs. Failed images can be inspected with `scripts/status.py --failed` and reset with `--reset-failed`.
- **Restartable discovery:** Running `discover.py` multiple times is safe — existing records are skipped.
- **Restartable processing:** Running `process.py` on an already-partially-processed dataset continues from where it left off.

---

## Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| Python over JavaScript | Tesseract bindings, OpenCV, scikit-image, and PIL have mature Python packages on conda-forge. Node.js OCR tooling is comparatively thin. |
| conda-forge over pip | Binary wheels for Tesseract, OpenCV, and scikit-image are reliably provided on conda-forge for Windows, eliminating compilation steps. |
| MSSQL as queue | SQL Server's `UPDATE … OUTPUT` makes atomic claiming straightforward. |
| SQLAlchemy 2.0 ORM | Type-safe models, connection pooling, and engine abstraction. The same models work against the test SQLite DB and production MSSQL without code changes. |
| Mirror output directory structure | Preserves the organisational logic of the input tree (which likely encodes date, batch, source, etc.) and makes it trivial to map a JSON document back to its source file and vice versa. |
| `full_text` as flat string in JSON | Elasticsearch performs best when the primary search field is a flat string. Per-page detail is available in the `pages` array for applications that need it. |
| No in-process image caching | With two million large TIFF files, an LRU cache would likely thrash. Each image is opened once, processed, and released. |

---

## Getting Started

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- SQL Server (any edition including Express) with a blank database created
- [ODBC Driver 17 or 18 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

### Installation

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate ocr-pipeline
```

### Database setup

Run `setup_db.sql` against your SQL Server instance in SSMS, Azure Data Studio, or sqlcmd:

```bash
sqlcmd -S localhost -d ocr_pipeline -E -i setup_db.sql
```

### Configuration

Edit `config.yaml`. At minimum, set:

```yaml
database:
  server: "YOUR_SERVER"
  database: "ocr_pipeline"

input:
  root_path: "C:\\path\\to\\tiff\\images"

output:
  root_path: "C:\\path\\to\\ocr\\output"
```

### Running

```bash
# Step 1: Register all images (safe to re-run)
python scripts/discover.py

# Step 2: Start OCR processing
python scripts/process.py --workers 8

# Step 3: Monitor progress (in a second terminal)
python scripts/status.py --watch
```

---

## Configuration Reference

See [config.yaml](config.yaml) — every setting is documented inline. Key sections:

| Section | Purpose |
|---|---|
| `database` | SQL Server connection details |
| `input` | Source TIFF directory and file extensions |
| `output` | Output directory and formats (txt, json) |
| `ocr` | Engine selection, Tesseract language and config string |
| `preprocessing` | Toggle and tune each preprocessing step |
| `pipeline` | Worker count, batch size, retry limits, log settings |

---

## CLI Reference

```bash
# Discover and register images
python scripts/discover.py [--config config.yaml] [--batch-size 500]

# Run the pipeline
python scripts/process.py [--config config.yaml] [--workers N]

# Show summary
python scripts/status.py [--config config.yaml]

# Auto-refresh summary every 30 seconds
python scripts/status.py --watch

# List permanently failed images
python scripts/status.py --failed

# Reset all failed images back to pending
python scripts/status.py --reset-failed
```

---

## Future Improvements

### GPU acceleration

The `EasyOCREngine` and `PaddleOCREngine` classes in `pipeline/ocr_engine.py` are already implemented and share the same interface as `TesseractEngine`. Switching is a one-line change in `config.yaml`:

```yaml
ocr:
  engine: "easyocr"   # or "paddleocr"
```

For multi-GPU servers, a worker could be pinned to a specific GPU by passing `CUDA_VISIBLE_DEVICES` as an environment variable per worker. This requires no changes to the pipeline logic.

### Elasticsearch ingestion service

The `.json` files produced by the pipeline are Elasticsearch-ready. A logical next step is a separate ingestion script or FastAPI background task that:

1. Watches `ocr_images` for newly `complete` rows
2. Reads the corresponding `.json` file
3. Bulk-indexes into Elasticsearch using the `elastic-transport` Python client

The `relative_path` field in each document provides a natural Elasticsearch document ID, enabling idempotent re-indexing.

### Layout analysis

Tesseract `--psm 3` treats each page as a single text block. For documents with complex layouts (multi-column newspapers, tables, forms), a dedicated layout analysis step using a tool like **Detectron2** (CPU-capable) or **LayoutParser** could identify regions of interest before OCR is applied per-region. This significantly improves accuracy on structured historical documents.

### Document classification

A lightweight classifier (e.g. a fine-tuned ResNet or a simple CNN trained on document thumbnails) could tag each image before OCR with its document type (letter, form, newspaper, photograph). This would allow different preprocessing profiles and Tesseract page segmentation modes to be applied per document type, and would enrich the Elasticsearch document with a filterable `document_type` field.

### Handwriting recognition

Tesseract performs poorly on handwritten text. If a proportion of the images contain handwriting, a dedicated HTR (Handwritten Text Recognition) model such as **TrOCR** (CPU-capable via `transformers`) or **Kraken** could be integrated as an additional engine option. A classification step (see above) could route handwritten pages to the HTR engine automatically.

### Confidence-based review queue

Images or pages where the OCR confidence score falls below a configurable threshold could be flagged in the database and surfaced in the web application for manual transcription or correction, creating a human-in-the-loop quality assurance workflow.

### Distributed processing across multiple machines

The current design scales vertically (more cores on one machine). Because the queue lives in MSSQL and workers are stateless, horizontal scaling requires no code changes: run `process.py` on additional machines pointing at the same `config.yaml` database and input/output paths (assuming shared network storage). Worker identity (`hostname-PID`) already disambiguates rows claimed by different machines.

### Incremental discovery with filesystem watcher

Replace the batch `discover.py` scan with a persistent filesystem watcher (e.g. `watchdog` on conda-forge) that enqueues new files as they are dropped into the input directory in near-real-time.
