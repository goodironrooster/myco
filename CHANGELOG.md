# Changelog

All notable changes to MYCO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## ⚠️ Development Status

**MYCO is designed for local use** with small LLM models (4B-9B parameters). The architecture was specifically crafted to work within small model constraints:

- Limited context windows → Layered context loading
- Lower reasoning capacity → Simple, focused tools
- Tendency to loop → Iteration budgets and loop detection
- No conversation history → Stigmergic memory in code

**Current Status:**
- Core features are operational and tested
- Primary testing with 4B and 9B models
- Limited testing with larger models
- **Needs more real-world testing** before production use

Contributions welcome: Testing reports, bug findings, and usage feedback highly valued.

---

## [Unreleased]

### Planned for v1.1
- VS Code extension
- Real-time entropy display
- Chat interface

---

## [1.0.0] - 2026-04-14

### Added

#### Core Features
- **Entropy Analysis** - Shannon entropy calculation for code modules
- **Entropy Regime Detection** - Crystallized/dissipative/diffuse classification
- **Entropy Budget Enforcement** - Blocks changes with ΔH > 0.15
- **Refactoring Suggestions** - Regime-based refactoring recommendations

#### Stigmergic Memory
- **File Annotations** - `# ⊕ H:0.73 | press:refactor | age:4` format
- **Cross-Session Learning** - Pattern library across sessions
- **Anti-Pattern Tracking** - Mistake tracking and warnings
- **Session Recording** - Automatic session outcome tracking

#### Autopoietic Gate
- **Entropy Delta Checking** - Blocks harmful changes
- **Tensegrity Validation** - Import graph structure validation
- **Import Boundary Enforcement** - Tension/compression boundary rules

#### Test Co-Creation
- **Auto Test Generation** - Tests created with every Python file
- **Test Stub Creation** - `tests/test_<module>.py` automatically
- **Immediate Test Execution** - Run tests after file creation
- **Coverage Tracking** - Track test creation per file

#### Quality Tracking
- **Per-File Metrics** - Quality tracking per file
- **Project Health Dashboard** - Overall health scoring (0.0-1.0)
- **Quality Trends** - Improving/stable/degrading classification
- **Attention Alerts** - Files needing attention

#### Agent Tools (24 total)
- File operations (read, write, edit, append, delete)
- Search tools (text, grep, definitions)
- Command execution (run_command, run_python, test_pytest)
- Architecture awareness (dependencies, dependents, affected files)
- Session memory (record, retrieve, learn)
- Quality feedback (trends, health, attention)

#### Verification & Certainty
- **Syntax Verification** - py_compile integration
- **Type Inference** - Detect type mismatches
- **Contract Generation** - Pre/post condition generation
- **Property Test Generation** - 7+ properties per file
- **Integration Verification** - Import/export matching

#### Performance & Reliability
- **Path Normalization** - Consistent path resolution
- **Action Memory** - Track last 20 actions
- **Loop Detection** - Detect repetitive patterns at count 3
- **Loop Recovery** - Category-specific recovery suggestions
- **Exploration Tracking** - Prevent endless exploration
- **Iteration Budget** - 30 max, warns at 75%

#### Project Context
- **World Model** - `.myco/world.json` persistence
- **Session Tracking** - 53+ sessions tracked
- **Health Scoring** - Project health calculation
- **Dependency Graph** - Import graph tracking

### Fixed

#### Critical Bugs
- **Path Normalization Bug** - Fixed path resolution for all file operations
- **Action Memory Bug** - Fixed `_iteration` attribute error
- **Loop Recovery Bug** - Improved recovery with category-specific suggestions
- **Import Error** - Added missing `re` import for search_definitions
- **run_python Paths** - Auto-prepends project root to sys.path

#### Minor Issues
- **Unicode Encoding** - Fixed summary file encoding errors
- **Syntax Warnings** - Fixed escape sequence warnings
- **Test Co-Creation** - Skip test files (don't create tests for tests)
- **Pytest Paths** - Fixed pytest working directory

### Changed

#### Performance Improvements
- **4B Model Support** - Optimized for Qwen3.5-4B (45-55 tok/s)
- **GPU Acceleration** - CUDA optimization
- **Memory Usage** - Reduced memory footprint
- **Action Tracking** - More efficient action memory

#### User Experience
- **Better Error Messages** - More actionable error messages
- **Progress Tracking** - Iteration warnings at 75%
- **Loop Detection** - Earlier detection (count 3 instead of 6)
- **Recovery Suggestions** - Category-specific suggestions

### Documentation

- **README.md** - Complete landing page
- **INSTALL.md** - Installation guide
- **CONTRIBUTING.md** - Contribution guidelines
- **QUICK_START.md** - Getting started guide
- **MYCO.md** - Vision and philosophy
- **PERFORMANCE_TESTING_GUIDE.md** - Testing documentation

### Technical

- **Tests** - 304 passing tests
- **Type Hints** - 100% type hint coverage
- **Docstrings** - 100% docstring coverage
- **CI/CD** - GitHub Actions setup
- **Code Quality** - Ruff, mypy, pytest integration

---

## [0.9.3] - 2026-03-26

### Added
- Phase 3: Certainty features
- Type inference
- Contract generation
- Property testing
- Integration verification

### Fixed
- Phase 2: Self-improvement features
- Session memory
- Quality feedback loops

---

## [0.9.0] - 2026-03-15

### Added
- Phase 2: Self-improvement features
- Session memory
- Pattern library
- Quality tracking

---

## [0.8.0] - 2026-03-01

### Added
- Phase 1: Sustainable coding features
- Entropy budgets
- Refactoring triggers
- Test co-creation
- Dependency tracking

---

## [0.7.0] - 2026-02-15

### Added
- Autopoietic gate
- Tensegrity validation
- Import boundary enforcement

---

## [0.6.0] - 2026-02-01

### Added
- Stigmergic annotations
- Cross-session learning
- World model

---

## [0.5.0] - 2026-01-15

### Added
- Entropy calculation
- Entropy regime detection
- Initial agent tools

---

## [0.1.0] - 2025-12-01

### Added
- Initial release
- Basic entropy tracking
- File editing tools
- Command execution

---

## Version Numbering

- **MAJOR.MINOR.PATCH**
- **MAJOR** - Breaking changes
- **MINOR** - New features (backwards compatible)
- **PATCH** - Bug fixes (backwards compatible)

## Release Schedule

- **Major releases** - Quarterly
- **Minor releases** - Monthly
- **Patch releases** - As needed

---

**Legend:**
- ✅ Added
- 🐛 Fixed
- 🔄 Changed
- 📝 Documentation
- ⚠️ Deprecated
- ❌ Removed
