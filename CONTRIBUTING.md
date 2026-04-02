# Contributing to MYCO

**Version:** 1.0.0
**Last Updated:** 2026-04-01

Thank you for your interest in contributing to MYCO! This document provides guidelines and instructions for contributing.

---

## 🍄 Philosophy

MYCO is not just another AI coding tool. It's a **thermodynamic coding agent** that optimizes for codebase health rather than task completion.

**Core Principles:**
1. **Substrate-First** - Code is a living substrate, not a product
2. **Stigmergy** - Memory through annotations, not databases
3. **Autopoiesis** - Self-maintaining, self-improving
4. **Local-Only** - Privacy-first, no cloud dependencies
5. **Entropy-Aware** - Prevent technical debt proactively

Before contributing, please read [MYCO.md](MYCO.md) to understand the vision.

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/myco.git
cd myco

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_USERNAME/myco.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_entropy.py

# Run with coverage
pytest --cov=myco tests/
```

---

## 📝 How to Contribute

### Reporting Bugs

**Before reporting:**
- Check existing issues
- Test on latest version
- Gather reproduction steps

**Bug report template:**
```markdown
**Description:** Clear description of the bug

**Reproduction:**
1. Step 1
2. Step 2
3. Error occurs

**Expected:** What should happen
**Actual:** What actually happened

**Environment:**
- OS: Windows 11
- Python: 3.11
- Model: Qwen3.5-4B
- MYCO version: 1.0.0

**Logs:** Attach relevant logs
```

### Suggesting Features

**Before suggesting:**
- Check roadmap
- Check existing feature requests
- Ensure it aligns with MYCO philosophy

**Feature request template:**
```markdown
**Problem:** What problem does this solve?

**Proposal:** Describe the feature

**Alignment:** How does this fit MYCO philosophy?

**Alternatives:** What other solutions exist?

**Impact:** Who benefits from this?
```

### Code Contributions

#### 1. Find an Issue

- Look for `good first issue` label
- Check `help wanted` issues
- Or propose your own fix

#### 2. Create Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/issue-123
```

#### 3. Make Changes

**Coding Standards:**
- Follow PEP 8
- Add type hints to all functions
- Add docstrings to all classes/methods
- Write tests for new features
- Keep functions small (<50 lines ideal)

**Example:**
```python
def calculate_entropy(modules: List[str]) -> float:
    """Calculate coupling entropy for module list.
    
    Args:
        modules: List of module paths to analyze
        
    Returns:
        Shannon entropy value (0.0-1.0)
        
    Raises:
        ValueError: If modules list is empty
    """
    if not modules:
        raise ValueError("Modules list cannot be empty")
    
    # Implementation here
    return entropy_value
```

#### 4. Write Tests

**Test structure:**
```python
def test_entropy_calculation():
    """Test entropy calculation for simple case."""
    # Arrange
    modules = ["module_a", "module_b"]
    
    # Act
    result = calculate_entropy(modules)
    
    # Assert
    assert 0.0 <= result <= 1.0
    assert isinstance(result, float)
```

**Run your tests:**
```bash
pytest tests/test_your_feature.py -v
```

#### 5. Commit Changes

**Commit message format:**
```
type(scope): brief description

Longer description if needed.

Fixes #123
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code restructuring
- `test` - Tests
- `chore` - Maintenance

**Example:**
```
feat(entropy): add regime detection

Implements entropy regime detection (crystallized/dissipative/diffuse).

Fixes #45
```

#### 6. Submit Pull Request

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open PR on GitHub:
   - Use PR template
   - Link related issues
   - Add screenshots if UI changes
   - Describe testing done

3. Respond to reviews:
   - Address feedback
   - Make requested changes
   - Keep discussion professional

---

## 🧪 Testing Guidelines

### Writing Tests

**Coverage requirements:**
- New features: 100% coverage required
- Bug fixes: Test the specific bug
- Refactors: Maintain existing coverage

**Test categories:**
```
tests/
├── unit/           # Unit tests
│   ├── test_entropy.py
│   ├── test_stigma.py
│   └── test_gate.py
├── integration/    # Integration tests
│   ├── test_agent.py
│   └── test_tools.py
└── performance/    # Performance tests
    └── test_speed.py
```

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest tests/unit/

# With coverage
pytest --cov=myco --cov-report=html

# Performance tests
pytest tests/performance/ --benchmark
```

---

## 📚 Documentation

### Documentation Standards

**All code needs:**
- Type hints
- Docstrings (Google style)
- Comments for complex logic
- Examples for public APIs

**Example docstring:**
```python
def check_entropy_budget(
    current: str,
    proposed: str,
    max_delta: float = 0.15
) -> Tuple[bool, float, float, str]:
    """Check if code change is within entropy budget.
    
    Args:
        current: Current file content
        proposed: Proposed new content
        max_delta: Maximum allowed entropy change
        
    Returns:
        Tuple of (within_budget, current_H, proposed_H, message)
        
    Example:
        >>> within, curr, prop, msg = check_entropy_budget(
        ...     "old code", "new code"
        ... )
        >>> within
        True
    """
```

### Documentation Types

**We need help with:**
- Tutorials (how-to guides)
- API reference
- Architecture documentation
- Performance optimization guides
- Troubleshooting guides

---

## 🎯 Areas We Need Help

### High Priority

1. **Documentation** 
   - Tutorials for beginners
   - API reference completion
   - Troubleshooting guide

2. **Testing**
   - More edge case tests
   - Performance benchmarks
   - Integration tests

3. **Examples**
   - Sample projects
   - Use case demonstrations
   - Before/after comparisons

### Medium Priority

4. **IDE Integration**
   - VS Code extension
   - JetBrains plugin
   - Neovim integration

5. **Performance**
   - Optimization for large codebases
   - Caching strategies
   - Parallel processing

6. **Model Optimization**
   - Fine-tuning datasets
   - Prompt engineering
   - Model comparisons

---

## 🔧 Development Workflow

### Before Submitting

**Checklist:**
- [ ] Tests pass (`pytest`)
- [ ] Type checking passes (`mypy myco/`)
- [ ] Linting passes (`ruff check myco/`)
- [ ] Documentation updated
- [ ] Commit messages follow format
- [ ] Branch is up to date with main

### Code Review Process

1. **Automated Checks**
   - CI runs tests
   - Type checking
   - Linting
   - Coverage check

2. **Human Review**
   - Maintainer reviews code
   - Provides feedback
   - Requests changes if needed

3. **Merge**
   - Squash and merge
   - Delete branch
   - Release notes updated

---

## 📖 Resources

### Documentation

- [MYCO.md](MYCO.md) - Vision & philosophy
- [INSTALL.md](INSTALL.md) - Installation guide
- [QUICK_START.md](QUICK_START.md) - Getting started
- [API Reference](docs/api.md) - API documentation

### External Resources

- [PEP 8](https://pep8.org/) - Python style guide
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pytest documentation](https://docs.pytest.org/)
- [Type hints guide](https://docs.python.org/3/library/typing.html)

---

## 🤝 Community Guidelines

### Be Respectful

- Treat all contributors with respect
- Focus on constructive criticism
- Assume good intentions
- Keep discussions professional

### Be Helpful

- Answer questions when you can
- Share your knowledge
- Mentor new contributors
- Document solutions

### Be Patient

- Review takes time
- Maintainers are volunteers
- Complex issues need discussion
- Not all PRs will be accepted

---

## 📜 License

By contributing to MYCO, you agree that your contributions will be licensed under the MIT License.

---

## 🎉 Thank You!

Every contribution helps make MYCO better. Whether it's a bug report, feature suggestion, documentation improvement, or code contribution - we appreciate your help!

**Together, we're building AI agents that write sustainable, self-improving code.** 🍄

---

**Questions?** Open an issue or join our discussions!
