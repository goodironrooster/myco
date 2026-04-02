# MYCO Quick Start Guide

**Get MYCO working in 5 minutes**

---

## Step 1: Install MYCO (2 minutes)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/myco.git
cd myco

# Install dependencies
pip install -e .
```

**Verify installation:**
```bash
python -c "import myco; print('MYCO installed!')"
```

---

## Step 2: Start Model Server (2 minutes)

MYCO needs a local LLM server. Choose one:

### Option A: LM Studio (Easiest)

1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install and launch
3. Download a model (Qwen3.5-4B recommended)
4. Click "Start Server" (port 1234)

### Option B: Ollama

```bash
# Install from ollama.ai
ollama pull qwen3.5:4b
ollama serve
```

### Option C: llama.cpp (Advanced)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make LLAMA_CUDA=1  # For NVIDIA GPU
./server -m your-model.gguf --port 1234
```

**Verify server is running:**
```bash
curl http://localhost:1234/health
# Should return: {"status":"ok"}
```

---

## Step 3: Run Your First Task (1 minute)

```bash
# Create a test project
mkdir my_test_project
cd my_test_project

# Run MYCO
myco "Create a simple calculator with addition and subtraction"
```

**Watch MYCO:**
1. Analyze the task
2. Create `calculator.py`
3. Create `tests/test_calculator.py`
4. Run tests
5. Show you the results

---

## Step 4: See the Results

```bash
# See what MYCO created
ls -la
# calculator.py  tests/

# View the code
cat calculator.py

# Run the tests
pytest tests/
```

---

## 🎯 What to Try Next

### Example Tasks

**Simple:**
```bash
myco "Add type hints to calculator.py"
myco "Add docstrings to all functions"
```

**Medium:**
```bash
myco "Add multiplication and division to calculator"
myco "Add input validation to all methods"
```

**Complex:**
```bash
myco "Create a REST API for the calculator"
myco "Add a command-line interface"
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
myco "Create user service" -v
# Shows detailed progress
```

---

## 🛠️ Common Issues

### "Connection refused"

**Problem:** Model server not running

**Solution:**
```bash
# Start server (LM Studio, Ollama, or llama.cpp)
# Then verify:
curl http://localhost:1234/health
```

### "Module not found: myco"

**Problem:** MYCO not installed

**Solution:**
```bash
cd myco
pip install -e .
```

### "Out of memory"

**Problem:** Model too large for your GPU

**Solution:**
- Use smaller model (4B instead of 9B)
- Reduce context size: `-c 16384`
- Close other GPU applications

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
myco "Create a todo list API with create, read, update, delete"
```

Happy coding! 🍄
