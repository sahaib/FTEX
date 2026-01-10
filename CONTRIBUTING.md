# Contributing to FTEX

Thank you for your interest in contributing to FTEX! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

---

## Getting Started

### Types of Contributions

We welcome:
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🧪 Test coverage
- 🎨 UI/UX improvements
- 🌐 Translations

### Issues

Before starting work:
1. Check existing issues to avoid duplicates
2. For bugs, create an issue with reproduction steps
3. For features, open a discussion first
4. Wait for maintainer feedback before large changes

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Ollama (optional, for AI features)

### Setup Steps

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ftex.git
cd ftex

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 5. Copy environment file
cp .env.example .env
# Edit .env with your Freshdesk credentials

# 6. Run tests
pytest

# 7. Create a branch for your changes
git checkout -b feature/your-feature-name
```

---

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-slack-integration`
- `fix/extraction-resume-bug`
- `docs/improve-readme`
- `refactor/cleanup-config`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(analysis): add sentiment analysis to deep AI
fix(extractor): handle rate limit errors gracefully
docs(readme): add troubleshooting section
```

---

## Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run checks locally**
   ```bash
   # Format code
   black .
   
   # Lint
   ruff check .
   
   # Type check
   mypy .
   
   # Run tests
   pytest
   ```

3. **Create Pull Request**
   - Fill out the PR template
   - Link related issues
   - Add screenshots for UI changes
   - Request review from maintainers

4. **Address feedback**
   - Respond to all comments
   - Push fixes as new commits
   - Re-request review when ready

5. **Merge**
   - Maintainer will squash and merge
   - Delete your branch after merge

---

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these tools:
- **Formatter**: Black (line length 100)
- **Linter**: Ruff
- **Type Checker**: mypy

```bash
# Format
black --line-length 100 .

# Lint
ruff check --fix .

# Type check
mypy src/
```

### Code Guidelines

```python
# Use type hints
def process_ticket(ticket: Dict[str, Any]) -> Optional[str]:
    ...

# Use dataclasses for data containers
@dataclass
class TicketStats:
    total: int
    resolved: int
    pending: int

# Use pathlib for file paths
from pathlib import Path
output_dir = Path("output")

# Use logging, not print
import logging
logger = logging.getLogger(__name__)
logger.info("Processing ticket %s", ticket_id)

# Use config module for settings
from config import config
url = config.freshdesk.base_url
```

### File Organization

```
src/
├── extraction/      # Freshdesk API scripts
├── analysis/        # AI analysis scripts
├── reports/         # Report generators
└── shared/          # Shared utilities
    ├── config.py
    ├── logging.py
    └── utils.py
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific file
pytest tests/test_extractor.py

# Specific test
pytest tests/test_extractor.py::test_rate_limiting
```

### Writing Tests

```python
# tests/test_extractor.py
import pytest
from src.extraction.freshdesk_extractor_v2 import FreshdeskExtractor

class TestFreshdeskExtractor:
    @pytest.fixture
    def extractor(self):
        return FreshdeskExtractor(api_key="test", domain="test")
    
    def test_rate_limiting(self, extractor):
        # Test rate limit handling
        ...
    
    def test_checkpoint_resume(self, extractor, tmp_path):
        # Test checkpoint functionality
        ...
```

### Test Coverage

Aim for:
- 80%+ coverage for new code
- 100% coverage for critical paths (extraction, analysis)
- Integration tests for API interactions

---

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def analyze_tickets(tickets: List[Dict], use_ai: bool = True) -> AnalysisResult:
    """Analyze support tickets and identify patterns.
    
    Args:
        tickets: List of ticket dictionaries from Freshdesk
        use_ai: Whether to use AI for deeper analysis
        
    Returns:
        AnalysisResult containing findings and recommendations
        
    Raises:
        ValueError: If tickets list is empty
        ConnectionError: If AI service is unavailable
        
    Example:
        >>> result = analyze_tickets(tickets, use_ai=True)
        >>> print(result.top_issues)
    """
```

### README Updates

When adding features:
1. Update feature list
2. Add usage examples
3. Update architecture diagram if needed
4. Add to troubleshooting if relevant

---

## Questions?

- 💬 Open a [Discussion](https://github.com/YOUR_ORG/ftex/discussions)
- 🐛 Report [Issues](https://github.com/YOUR_ORG/ftex/issues)
- 📧 Email: your-email@example.com

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
