# Contributing to MonitaQC

Thank you for your interest in contributing to MonitaQC!

## Development Setup

1. Clone the repository
2. Install Docker and Docker Compose
3. Set up NVIDIA Container Toolkit for GPU support
4. Copy `.env.sample` to `.env` and configure

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

Before submitting changes:

1. Test locally with `docker compose up`
2. Verify all services start correctly
3. Test camera capture and YOLO inference
4. Check Redis connectivity
5. Verify database migrations

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test thoroughly
4. Commit with clear messages
5. Push to your branch
6. Create a merge request

## Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat: Add multi-language OCR support

- Implemented EasyOCR integration
- Added language detection
- Updated documentation

Closes #123
```

## Questions?

Contact: contact@virasad.ir
