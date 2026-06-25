-- OCR Pipeline Database Schema for SQLite

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Images table
-- Tracks every TIFF file discovered on the filesystem and its processing state.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ocr_images (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT NOT NULL UNIQUE,  -- Absolute path on server
    file_name        TEXT NOT NULL,
    file_size_bytes  INTEGER,
    status           TEXT NOT NULL DEFAULT 'pending',
    worker_id        TEXT,
    started_at       DATETIME,
    completed_at     DATETIME,
    error_message    TEXT,
    retry_count      INTEGER NOT NULL DEFAULT 0,
    output_path      TEXT,
    page_count       INTEGER,
    created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for status and retry_count (used by worker claim query)
CREATE INDEX IF NOT EXISTS IX_ocr_images_status_retry
    ON ocr_images (status, retry_count, id, file_path, file_name);

-- Index for status and started_at (used by stale-recovery query)
CREATE INDEX IF NOT EXISTS IX_ocr_images_processing_started
    ON ocr_images (status, started_at);

-- ---------------------------------------------------------------------------
-- Results table
-- Stores the extracted text and metadata for each page of each image.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ocr_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL,
    page_number         INTEGER NOT NULL DEFAULT 1,
    extracted_text      TEXT,
    confidence_score    REAL,
    processing_time_ms  INTEGER,
    ocr_engine          TEXT,
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(image_id, page_number),
    FOREIGN KEY (image_id) REFERENCES ocr_images(id) ON DELETE CASCADE
);

-- Index for image_id
CREATE INDEX IF NOT EXISTS IX_ocr_results_image_id ON ocr_results (image_id);

-- ---------------------------------------------------------------------------
-- Convenience view: pipeline summary
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_pipeline_summary;
CREATE VIEW vw_pipeline_summary AS
SELECT
    status,
    COUNT(*) AS image_count,
    SUM(file_size_bytes) / 1073741824.0 AS total_size_gb,
    AVG((julianday(completed_at) - julianday(started_at)) * 86400000.0) AS avg_processing_ms
FROM ocr_images
GROUP BY status;