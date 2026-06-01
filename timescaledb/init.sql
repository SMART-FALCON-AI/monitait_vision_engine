-- TimescaleDB initialization script for MonitaQC
-- This runs automatically on first container startup

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Production metrics hypertable
CREATE TABLE IF NOT EXISTS production_metrics (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    encoder_value INTEGER,
    ok_counter INTEGER,
    ng_counter INTEGER,
    shipment TEXT,
    is_moving BOOLEAN,
    downtime_seconds INTEGER
);

SELECT create_hypertable('production_metrics', 'time', if_not_exists => TRUE);

-- Inference results hypertable
CREATE TABLE IF NOT EXISTS inference_results (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    shipment TEXT,
    image_path TEXT,
    detections JSONB,
    detection_count INTEGER,
    inference_time_ms INTEGER,
    model_used TEXT,
    pipeline_name TEXT,
    module_id TEXT,
    phase_id INTEGER,
    encoder_value BIGINT   -- capture-time encoder (3.21.0) — roll-position charts
);

SELECT create_hypertable('inference_results', 'time', if_not_exists => TRUE);

-- Ejection events hypertable (3.17.0) — one row per triggered procedure with Store=ON
CREATE TABLE IF NOT EXISTS ejection_events (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    shipment TEXT,
    procedure_name TEXT,
    reason TEXT,
    encoder_value BIGINT
);

SELECT create_hypertable('ejection_events', 'time', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_production_shipment ON production_metrics (shipment, time DESC);
CREATE INDEX IF NOT EXISTS idx_inference_shipment ON inference_results (shipment, time DESC);
CREATE INDEX IF NOT EXISTS idx_production_time ON production_metrics (time DESC);
CREATE INDEX IF NOT EXISTS idx_inference_time ON inference_results (time DESC);
CREATE INDEX IF NOT EXISTS idx_ejection_time ON ejection_events (time DESC);
CREATE INDEX IF NOT EXISTS idx_ejection_proc ON ejection_events (procedure_name, time DESC);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO monitaqc;
