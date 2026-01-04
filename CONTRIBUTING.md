# Contributing to VoiceNotes

Thank you for your interest in contributing to VoiceNotes! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/VoiceNotes.git`
3. Create a conda environment: `conda env create -f environment.yml`
4. Activate the environment: `conda activate voice`
5. Install development dependencies: `pip install -e ".[dev]"`
6. Install pre-commit hooks: `pre-commit install`

## Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Run `black .` and `isort .` before committing

## Testing

- Write tests for new features
- Ensure all tests pass: `pytest`
- Maintain test coverage above 80%

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run linting: `flake8 .`
5. Commit your changes: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Create a Pull Request

## Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers if applicable
- Follow conventional commit format when possible

## Questions?

Feel free to open an issue for any questions or clarifications.

