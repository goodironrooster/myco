# MYCO Quick Start Guide

**Get MYCO working in 5 minutes**

---

## Step 1: Install MYCO (1 minute)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/myco.git
cd myco

# Install dependencies
pip install -e .
```

**Verify installation:**
```bash
python -c "from cli.main import myco; print('MYCO installed!')"
```

---

## Step 2: Start MYCO (30 seconds)

MYCO auto-starts everything — CUDA GPU server, model loading, 200K context window.

```bash
python -m cli.main
```

That's it. MYCO will:
1. Find `llama-server.exe` (looks in `llama-cpp/` and common paths)
2. Load the `Qwen3.5-9B-Q4_0.gguf` model from the project root
3. Start the server with **CUDA GPU** (99 layers), **200K context**, **flash attention**
4. Drop you into the interactive agent with 42 tools

**Verify it's working:**
```
✓ Server ready at http://127.0.0.1:1234
✓ MYCO Vision enabled
  Tools Available: 42
You>
```

---

## Step 3: Run Your First Task

In the interactive prompt, just type your task:

```
You> Create a simple calculator with addition and subtraction
```

**Watch MYCO:**
1. Analyze the task
2. Create `calculator.py`
3. Create `tests/test_calculator.py`
4. Run tests
5. Show you the results

Type `exit` when done.

---

## 🎯 What to Try Next

### Example Tasks (type these in the interactive prompt)

**Simple:**
```
Add type hints to calculator.py
Add docstrings to all functions
```

**Medium:**
```
Add multiplication and division to calculator
Add input validation to all methods
```

**Complex:**
```
Create a REST API for the calculator
Add a command-line interface
```

---

## 💡 Tips for Best Results

### 1. Be Specific

❌ **Bad:** "Make it better"
✅ **Good:** "Add error handling for division by zero"

### 2. Start Small

❌ **Bad:** "Build a complete e-commerce platform"
✅ **Good:** "Create a Product model with name and price"

### 3. Review Code

MYCO writes quality code, but always review:
- Check the logic
- Run the tests
- Verify it meets your needs

### 4. Use Verbose Mode

```bash
python -m cli.main -v
# Shows detailed server startup and progress
```

### 5. Custom Context Window

```bash
python -m cli.main -c 65536
# Use 64K context instead of default 200K
```

---

## 🛠️ Common Issues

### "llama-server.exe not found"

**Problem:** llama.cpp not installed

**Solution:**
```bash
# Download llama.cpp release with CUDA support
# Place llama-server.exe in D:\MYCO\llama-cpp\
# Or add it to your PATH
```

### "No .gguf model found"

**Problem:** Model file missing

**Solution:**
- Place a `.gguf` model in `D:\MYCO\` (default: `Qwen3.5-9B-Q4_0.gguf`)
- Or put any `.gguf` file in the current directory

### "Out of memory"

**Problem:** Model too large for your GPU

**Solution:**
- Use smaller model (4B instead of 9B)
- Reduce context size: `python -m cli.main -c 32768`
- Close other GPU applications

### "Connection refused" (server already running)

**Problem:** Old server instance on wrong port

**Solution:**
```bash
# Kill existing server
taskkill /F /IM llama-server.exe
# Then restart
python -m cli.main
```

---

## 📚 Next Steps

### Learn More

- [Installation Guide](INSTALL.md) - Detailed setup
- [Examples](examples/) - Sample projects
- [Technical Docs](docs/TECHNICAL.md) - How MYCO works

### Join the Community

- **GitHub Issues** - Report bugs, request features
- **Discussions** - Ask questions
- **Twitter** - Follow updates

---

## 🎉 You're Ready!

MYCO is now ready to help you write better code.

**Try this now:**
```bash
python -m cli.main
```

Then type:
```
Create a todo list API with create, read, update, delete
```

Happy coding! 🍄
