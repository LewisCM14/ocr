-- =============================================================================
-- OCR Pipeline Database Schema
-- Run this once against your target MSSQL database before starting the pipeline.
-- Compatible with SQL Server 2016+ and Azure SQL Database.
-- =============================================================================

USE ocr_pipeline;
GO

-- ---------------------------------------------------------------------------
-- Images table
-- Tracks every TIFF file discovered on the filesystem and its processing state.
--
-- Status values:
--   pending    - waiting to be processed (also set on eligible retries)
--   processing - currently being worked on by a worker process
--   complete   - OCR finished successfully
--   failed     - permanently failed (retry_count >= max_retries)
-- ---------------------------------------------------------------------------
IF OBJECT_ID('dbo.ocr_images', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_images (
        id               BIGINT IDENTITY(1,1)  NOT NULL,
        file_path        NVARCHAR(1000)         NOT NULL,  -- Absolute path on server
        file_name        NVARCHAR(255)          NOT NULL,
        file_size_bytes  BIGINT                 NULL,
        status           VARCHAR(20)            NOT NULL  DEFAULT 'pending',
        worker_id        VARCHAR(100)           NULL,      -- hostname + PID of processing worker
        started_at       DATETIME2              NULL,
        completed_at     DATETIME2              NULL,
        error_message    NVARCHAR(MAX)          NULL,
        retry_count      INT                    NOT NULL  DEFAULT 0,
        output_path      NVARCHAR(1000)         NULL,      -- Base path of output files (no extension)
        page_count       INT                    NULL,
        created_at       DATETIME2              NOT NULL  DEFAULT GETUTCDATE(),

        CONSTRAINT PK_ocr_images PRIMARY KEY CLUSTERED (id),
        CONSTRAINT UQ_ocr_images_file_path UNIQUE NONCLUSTERED (file_path)
    );
END
GO

-- Index used by the worker claim query (hot path)
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.ocr_images') AND name = 'IX_ocr_images_status_retry')
BEGIN
    CREATE INDEX IX_ocr_images_status_retry
        ON dbo.ocr_images (status, retry_count)
        INCLUDE (id, file_path, file_name);
END
GO

-- Index used by the stale-recovery query
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.ocr_images') AND name = 'IX_ocr_images_processing_started')
BEGIN
    CREATE INDEX IX_ocr_images_processing_started
        ON dbo.ocr_images (status, started_at)
        WHERE status = 'processing';
END
GO

-- ---------------------------------------------------------------------------
-- Results table
-- Stores the extracted text and metadata for each page of each image.
-- One row per (image, page) pair.
-- ---------------------------------------------------------------------------
IF OBJECT_ID('dbo.ocr_results', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_results (
        id                  BIGINT IDENTITY(1,1)  NOT NULL,
        image_id            BIGINT                NOT NULL,
        page_number         INT                   NOT NULL  DEFAULT 1,
        extracted_text      NVARCHAR(MAX)         NULL,
        confidence_score    FLOAT                 NULL,      -- 0.0 – 1.0
        processing_time_ms  INT                   NULL,
        ocr_engine          VARCHAR(50)           NULL,
        created_at          DATETIME2             NOT NULL  DEFAULT GETUTCDATE(),

        CONSTRAINT PK_ocr_results PRIMARY KEY CLUSTERED (id),
        CONSTRAINT FK_ocr_results_image FOREIGN KEY (image_id) REFERENCES dbo.ocr_images (id),
        CONSTRAINT UQ_ocr_results_image_page UNIQUE NONCLUSTERED (image_id, page_number)
    );
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.ocr_results') AND name = 'IX_ocr_results_image_id')
BEGIN
    CREATE INDEX IX_ocr_results_image_id ON dbo.ocr_results (image_id);
END
GO

-- ---------------------------------------------------------------------------
-- Convenience view: pipeline summary
-- ---------------------------------------------------------------------------
IF OBJECT_ID('dbo.vw_pipeline_summary', 'V') IS NOT NULL DROP VIEW dbo.vw_pipeline_summary;
GO
CREATE VIEW dbo.vw_pipeline_summary AS
SELECT
    status,
    COUNT(*)                                        AS image_count,
    SUM(file_size_bytes) / 1073741824.0             AS total_size_gb,
    AVG(DATEDIFF(MILLISECOND, started_at, completed_at)) AS avg_processing_ms
FROM dbo.ocr_images
GROUP BY status;
GO
