# Changelog

All notable changes to MonitaQC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Merge with fabric inspection capabilities (from FabriQC)
- Merge with signal counting capabilities (from PartQC Signal Counter)
- Unified admin interface for all QC modes
- Multi-application mode support
- Enhanced API with OpenAPI documentation
- Advanced analytics and reporting
- Cloud synchronization improvements

## [1.2.0] - 2025-12-29

### Changed
- **Project Reorganization**: Moved core service files to dedicated `vision_engine/` directory
- **Service Renamed**: `monitaqc_counter` → `monitaqc_vision` (better reflects functionality)
- Updated docker-compose.yml build context to `./vision_engine`
- Cleaner main directory structure with only service folders

### Added
- `vision_engine/README.md` - Comprehensive vision engine documentation
- `vision_engine/.dockerignore` - Docker build optimization
- Better name reflects computer vision and AI processing capabilities

### Improved
- Main directory is now less crowded with only service directories
- Better separation of concerns for each microservice
- Easier navigation and maintenance
- More descriptive service naming (vision_engine vs generic counter)

## [1.1.0] - 2025-12-29

### Changed
- **Lightweight Architecture**: Reduced from 11+ containers to 4 core services
- Optimized Redis with 256MB memory limit and LRU eviction policy
- Reduced logging overhead (5-10MB max per service vs 10MB)
- Simplified docker-compose.yml to essential services only
- Changed Redis from `redis:latest` to `redis:7-alpine` for smaller image size
- Reduced YOLO shared memory from 2GB to 1GB
- Removed YOLO replicas (deploy.replicas: 2 → 1) for simpler deployment

### Added
- `docker-compose.full.yml` with all optional services
- Cleanup service now included in lightweight mode (prevents disk full)
- Lightweight/Full mode documentation in README
- Resource comparison table in README

### Removed
- Shipment fulfillment web interface (moved to full mode)
- PostgreSQL database (moved to full mode)
- Celery workers (2 workers + beat moved to full mode)
- Video streaming service (moved to full mode)
- Image gallery service (moved to full mode)
- Port 554 from counter service (unused RTSP port)
- Dataset volume mount from YOLO (unnecessary in production)

### Fixed
- Container naming consistency across all services
- Network configuration simplified to bridge driver

## [1.0.0] - 2025-12-29

### Added
- Initial fork from PartQC Box Counter
- Comprehensive README with full documentation
- MonitaQC branding and naming
- Standardized docker-compose container names
- .gitignore for project files
- CONTRIBUTING.md guidelines
- This CHANGELOG.md file

### Changed
- Renamed all container services with `monitaqc_` prefix
- Updated documentation to reflect MonitaQC platform vision
- Improved README structure and clarity

### Project Status
- Forked from PartQC Box Counter
- Foundation for unified quality control platform
- Active development phase

---

## Version History

- **1.0.0** - Initial MonitaQC release (fork from PartQC Box Counter)
