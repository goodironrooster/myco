# MYCO - AI Coding Agent That Prevents Technical Debt

> **"Finally, an AI that cares about code quality as much as I do."**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-304%20passing-green)]()

**MYCO** is an AI coding agent that optimizes for **long-term codebase health**, not just task completion.

Unlike other AI tools that generate code without thinking about consequences, MYCO:
- 🛡️ **Prevents technical debt** before it happens
- ✅ **Auto-generates tests** for every file
- 🧠 **Learns from your sessions** and gets smarter
- 📊 **Tracks code quality** metrics in real-time
- 🚫 **Blocks harmful changes** automatically

---

## ⚡ Quick Start (60 seconds)

```bash
# Install
pip install -e .

# Start (auto-starts CUDA server with 200K context)
python -m cli.main
```

That's it. MYCO handles the rest.

---

## 🎯 Why MYCO?

### The Problem with Other AI Coding Tools

```
You: "Add a feature"
Other AI: *writes code quickly*
You: "But this creates technical debt..."
Other AI: "Not my problem"
Result: Code works today, breaks in 6 months
```

Sound familiar? You're not alone. Most AI coding assistants:
- ❌ Don't write tests
- ❌ Create messy code
- ❌ Don't learn from mistakes
- ❌ Make your codebase worse over time

### The MYCO Difference

```
You: "Add a feature"
MYCO: *analyzes code health first*
MYCO: "Wait, this approach creates technical debt. Here's a better way..."
MYCO: *writes code + tests + prevents future issues*
Result: Code works today AND in 6 months
```

---

## 🔥 Key Features

### 1. Technical Debt Prevention 🛡️

MYCO analyzes code complexity **before** making changes. If a change would make code worse, it suggests a better approach.

**Example:**
```
⚠️  This change would create a "god class" (too many responsibilities)
💡  Suggestion: Split into 3 focused classes instead

✓ Code health maintained
✓ Future maintenance easier
✓ Technical debt prevented
```

### 2. Automatic Test Generation ✅

Every file MYCO touches gets tests. Automatically. No reminders needed.

**What you get:**
- Test file created automatically
- Test stubs for all functions
- Tests run immediately
- Coverage tracked

### 3. Session Memory 🧠

MYCO remembers what it learned in previous sessions. It doesn't repeat the same mistakes.

**Benefits:**
- Learns your coding style
- Remembers successful patterns
- Avoids past mistakes
- Gets smarter over time

### 4. Code Quality Tracking 📊

Real-time health scores show if your codebase is improving or degrading.

**Metrics tracked:**
- Code complexity
- Test coverage
- Dependency health
- Quality trends


---

## 🚀 Try It Now

### Installation

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/myco.git
cd myco
pip install -e .

# Start MYCO (auto-starts CUDA server, loads model, enters interactive mode)
python -m cli.main
```

### Example Tasks

Once in interactive mode, just type your task:

```
# Simple
Add type hints to all functions in src/

# Medium
Create authentication service with JWT tokens

# Complex
Build a complete CRUD module with API, service, and tests
```

---

## 📖 What's Under the Hood

MYCO uses advanced techniques to maintain code quality:

- **Entropy Analysis** - Measures code complexity
- **Stigmergic Memory** - Leaves traces in code for future sessions
- **Autopoietic Gate** - Blocks changes that would degrade code
- **Dependency Tracking** - Knows what breaks when you change code

*Want to learn more?* Check out our [Technical Documentation](docs/TECHNICAL.md)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Overview & quick start |
| [INSTALL.md](INSTALL.md) | Installation guide |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute tutorial |
| [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) | **Important: Local-first design & limitations** |
| [Examples](examples/) | Example projects |
| [Contributing](CONTRIBUTING.md) | How to contribute |

---

## 🤝 Contributing

Contributions welcome! Whether it's:
- Bug reports
- Feature requests
- Documentation improvements
- Code contributions

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT License - Free for personal and commercial use.

---

## ⚠️ Development Status & Limitations

**MYCO was built for local use** and has been primarily tested with **small local LLM models** (4B-9B parameters). Many architectural decisions were made specifically to work within the constraints of small models:

- **Layered context loading** - Avoids overwhelming limited context windows
- **Focused agent tools** - Simple, single-purpose tools instead of complex multi-step operations
- **Entropy budgets** - Prevents small models from generating overly complex code
- **Stigmergic annotations** - External memory to compensate for no conversation history
- **Iteration limits** - Prevents small models from getting stuck in loops

**What this means for you:**

| Aspect | Status |
|--------|--------|
| **Core Features** | ✅ Operational |
| **Small Model Support** | ✅ Tested (4B-9B) |
| **Large Model Support** | ⚠️ Limited testing |
| **Production Use** | ⚠️ Needs more testing |
| **Edge Cases** | ⚠️ May encounter untested scenarios |

**If you plan to use MYCO:**

- ✅ Great for: Local development, learning, experimentation
- ⚠️ Use caution: Critical production workflows without additional testing
- 🔍 Help needed: More testing with diverse codebases and model sizes

**Contributions welcome:** Testing reports, bug findings, and real-world usage feedback are highly valued.

---

## 💬 Join the Community

- **GitHub Issues** - Report bugs, request features
- **Discussions** - Ask questions, share tips
- **Twitter** - Follow for updates

---

**Tired of AI-generated technical debt?** Give MYCO a try.
