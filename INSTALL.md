# MYCO Installation Guide

**Version:** 1.0.0
**Last Updated:** 2026-04-01

---

## ⚠️ Important Note: Local-First Design

**MYCO was built for local use** with small LLM models (4B-9B parameters). The entire architecture was designed around the constraints and capabilities of small local models:

- **Context efficiency** - Layered loading to avoid overwhelming small context windows
- **Tool design** - Simple, focused tools that small models can execute reliably
- **Safety mechanisms** - Entropy budgets and iteration limits prevent small models from going off-track
- **Memory strategy** - Stigmergic annotations compensate for lack of conversation history

**Testing Status:**
- ✅ Core features operational
- ✅ Tested with 4B and 9B models
- ⚠️ Limited testing with larger models (30B+)
- ⚠️ Needs more real-world testing before production use

---

## System Requirements

### Minimum Requirements

- **OS:** Windows 10/11, Linux, or macOS
- **Python:** 3.9 or higher
- **RAM:** 8GB (16GB recommended)
- **Storage:** 10GB free space

### Recommended Requirements

- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **CUDA:** 12.5 (for GPU acceleration)
- **RAM:** 32GB
- **Storage:** SSD with 20GB free space

---

## Step 1: Install Python

### Windows

1. Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Click "Install Now"

Verify installation:
```bash
python --version
# Should show: Python 3.9.x or higher
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### macOS

```bash
brew install python@3.9
```

---

## Step 2: Install CUDA (Optional but Recommended)

### Windows

1. Download CUDA 12.5 from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Run installer
3. Restart computer

Verify installation:
```bash
nvcc --version
nvidia-smi
```

### Linux

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-12-5
```

---

## Step 3: Download LLM Model

### Recommended Models

| Model | Size | Speed | Quality | Download |
|-------|------|-------|---------|----------|
| **Qwen3.5-4B-Uncensored** | 2.5 GB | 45-55 tok/s | Good | [HuggingFace](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-Aggressive-GGUF) |
| Qwen3.5-9B | 5.5 GB | 20-25 tok/s | Better | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B-Instruct-GGUF) |

### Download Instructions

1. Go to HuggingFace model page
2. Click "Files and versions"
3. Download the `.gguf` file (Q4_K_M quantization recommended)
4. Save to: `D:\LLM models\` (Windows) or `~/models/` (Linux/Mac)

---

## Step 4: Install LLM Server

### Option A: LM Studio (Easiest)

1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install and launch
3. Click "Download" tab
4. Search for "Qwen3.5-4B"
5. Click Download
6. Go to "Local Server" tab
7. Select downloaded model
8. Click "Start Server" (port 1234)

### Option B: llama.cpp (Advanced)

```bash
# Clone repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA
make clean
make LLAMA_CUDA=1 -j

# Run server
./server -m ../models/Qwen3.5-4B-Uncensored.gguf --port 1234 -c 32768
```

---

## Step 5: Install MYCO

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/myco.git
cd myco
```

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install MYCO
pip install -e .
```

### Verify Installation

```bash
python -c "from myco.entropy import calculate_entropy_from_content; print('MYCO installed successfully!')"
```

---

## Step 6: Configure MYCO

### Create Configuration File

Create `.myco/config.json` in your project directory:

```json
{
  "model": "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
  "base_url": "http://localhost:1234",
  "max_iterations": 30,
  "require_approval": false,
  "gpu_layers": 99,
  "context_size": 32768
}
```

### Set Project Root

MYCO automatically detects project root. Ensure your project has:
- `.myco/` directory (created automatically)
- Python files to analyze

---

## Step 7: Test Installation

### Quick Test

```bash
# Start model server (if not running)
# LM Studio: Click "Start Server"
# llama.cpp: Run ./server command

# Test MYCO
cd your_project_folder
python -m myco "List all Python files in this directory"
```

### Expected Output

```
✓ Agent initialized
✓ Project context loaded
✓ World model: 0 sessions

Task: List all Python files

[Agent works...]

✓ Task complete
Files found: 5
```

---

## Troubleshooting

### Issue: "Module not found: myco"

**Solution:**
```bash
# Ensure you're in the myco directory
cd myco

# Reinstall in development mode
pip install -e .
```

### Issue: "CUDA not available"

**Solution:**
1. Verify CUDA installation: `nvcc --version`
2. Check GPU: `nvidia-smi`
3. Reinstall llama.cpp with CUDA: `make LLAMA_CUDA=1`

### Issue: "Connection refused on port 1234"

**Solution:**
1. Start model server (LM Studio or llama.cpp)
2. Verify server is running: `curl http://localhost:1234/health`
3. Check firewall settings

### Issue: "Out of memory"

**Solution:**
1. Close other GPU applications
2. Use smaller model (4B instead of 9B)
3. Reduce context size: `-c 16384`
4. Reduce batch size: `-b 128`

### Issue: "Slow token generation (<10 tok/s)"

**Solution:**
1. Verify GPU is being used: `nvidia-smi`
2. Check GPU utilization in task manager
3. Ensure all layers on GPU: `-ngl 99`
4. Enable flash attention: `-fa on`

---

## Performance Optimization

### For Maximum Speed

1. **Use 4B model** - 2x faster than 9B
2. **Enable CUDA** - 20x faster than CPU
3. **Maximize GPU layers** - `-ngl 99`
4. **Enable flash attention** - `-fa on`
5. **Use SSD** - Faster model loading

### For Maximum Quality

1. **Use 9B model** - Better reasoning
2. **Increase context** - `-c 65536`
3. **Higher iterations** - `max_iterations: 50`
4. **Enable approval** - `require_approval: true`

---

## Next Steps

After installation:

1. **Read [QUICK_START.md](QUICK_START.md)** - Learn basic usage
2. **Read [MYCO.md](MYCO.md)** - Understand philosophy
3. **Run first task** - `python -m myco "Add type hints to src/"`
4. **Check performance** - `python -m myco report`

---

## Getting Help

- **Documentation:** See `docs/` folder
- **Issues:** Open GitHub issue
- **Discussions:** GitHub Discussions tab
- **Examples:** See `examples/` folder

---

**Installation Complete!** 🍄

Next: [Quick Start Guide](QUICK_START.md)
