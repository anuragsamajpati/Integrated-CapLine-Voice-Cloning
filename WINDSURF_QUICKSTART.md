# 🌊 Windsurf Quick Start Guide
## Integrating CapLine_TazzoX + Voice Cloning

---

## 🎯 What You'll Build

A unified system that:
- ✅ Translates speech in real-time (19 languages)
- ✅ Preserves the speaker's voice characteristics
- ✅ Combines Whisper ASR + M2M100 + AKF Voice Cloning
- ✅ Works through a web interface

---

## 📋 Prerequisites

Before starting in Windsurf:

1. **Python 3.8+** installed
2. **Your existing code**:
   - CapLine_TazzoX project folder
   - Voice Cloning project folder
3. **Windsurf IDE** installed

---

## 🚀 Step-by-Step in Windsurf

### Step 1: Create New Project

1. Open **Windsurf**
2. Click **File → Open Folder**
3. Create a new folder: `integrated-speech-translation`
4. Select and open this folder

### Step 2: Open Windsurf Terminal

- Press **Ctrl+`** (Windows/Linux) or **Cmd+`** (Mac)
- Or: **View → Terminal**

### Step 3: Run Setup Script

In the Windsurf terminal:

```bash
# Download the setup script (or create it manually)
# Make it executable
chmod +x setup_integration.sh

# Run it
./setup_integration.sh
```

**What this does:**
- Creates project structure
- Sets up virtual environment
- Installs all dependencies
- Creates necessary files

### Step 4: Copy Your Existing Code

**Using Windsurf File Explorer (Left sidebar):**

1. Right-click `backend/capline` → **Reveal in Finder/Explorer**
2. Copy your CapLine_TazzoX files here:
   - `app.py`
   - `requirements.txt`
   - Any other files

3. Right-click `backend/voice_cloning` → **Reveal in Finder/Explorer**
4. Copy your Voice Cloning files here:
   - `AKF_base_algo_full.py`
   - `akf_dashboard.py`
   - `akf_realtime_ui.py`

**Or use terminal:**

```bash
# From Windsurf terminal
cp -r ~/path/to/capline_project/backend/* backend/capline/
cp ~/path/to/Voice\ Cloning/*.py backend/voice_cloning/
```

### Step 5: Create Integration Files

**Create these 3 files in Windsurf:**

#### File 1: `backend/integration/feature_bridge.py`

1. In Windsurf: **File → New File**
2. Save as: `backend/integration/feature_bridge.py`
3. Paste the code from the integration guide

#### File 2: `backend/integration/integrated_pipeline.py`

1. **File → New File**
2. Save as: `backend/integration/integrated_pipeline.py`
3. Paste the code from the integration guide

#### File 3: `backend/integration/api_server.py`

1. **File → New File**
2. Save as: `backend/integration/api_server.py`
3. Paste the code from the integration guide

### Step 6: Install Dependencies

In Windsurf terminal:

```bash
# Make sure venv is activated (you should see (venv) in terminal)
# If not:
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r backend/requirements_integrated.txt
```

### Step 7: Test the Integration

In Windsurf terminal:

```bash
# Navigate to backend
cd backend

# Run the API server
python integration/api_server.py
```

**Expected output:**
```
🚀 Starting Integrated API Server...
📥 Loading Whisper ASR model...
🌍 Loading M2M100 translation model...
✅ Initialization complete!

 * Running on http://0.0.0.0:5000
```

### Step 8: Open Frontend

**Split terminal in Windsurf:**

1. Click the **split icon** in terminal (top right)
2. In the new terminal:

```bash
cd frontend
python -m http.server 8000
```

### Step 9: Test in Browser

1. Open browser: `http://localhost:8000`
2. Allow microphone access
3. Select target language (e.g., "Hindi")
4. Click "Start Recording" and speak
5. Click "Stop Recording"
6. Wait for processing
7. Listen to translated audio with your voice!

---

## 🎨 Windsurf Features to Use

### 1. **AI Code Completion**
- Windsurf will suggest code as you type
- Press **Tab** to accept suggestions

### 2. **Multi-File Editing**
- Open multiple files: **Ctrl+P** (Cmd+P on Mac)
- Type filename and press Enter
- Arrange in split view

### 3. **Integrated Terminal**
- Run backend and frontend in split terminals
- No need to switch windows

### 4. **Git Integration**
- Initialize git: Click **Source Control** icon (left sidebar)
- Click "Initialize Repository"
- Make commits as you progress

### 5. **Debugging**
- Set breakpoints: Click left of line numbers
- Press **F5** to start debugging
- Inspect variables in real-time

---

## 📁 Final Project Structure in Windsurf

```
integrated-speech-translation/      ← Your Windsurf workspace
├── backend/
│   ├── capline/                    ← Your CapLine files
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── voice_cloning/              ← Your Voice Cloning files
│   │   ├── AKF_base_algo_full.py
│   │   ├── akf_dashboard.py
│   │   └── akf_realtime_ui.py
│   ├── integration/                ← NEW: Integration layer
│   │   ├── __init__.py
│   │   ├── feature_bridge.py       ← Create this
│   │   ├── integrated_pipeline.py  ← Create this
│   │   └── api_server.py           ← Create this
│   └── requirements_integrated.txt
├── frontend/
│   ├── index.html
│   └── app.js
├── models/
├── outputs/
├── tests/
└── venv/
```

---

## 🔧 Using Windsurf AI Assistant

Windsurf has built-in AI assistance:

1. **Ask Questions:**
   - Press **Ctrl+L** (Cmd+L on Mac)
   - Type: "How do I modify the translation pipeline?"
   - Get AI-powered answers

2. **Generate Code:**
   - Highlight code
   - Right-click → "Ask AI to..."
   - Request modifications

3. **Debug Issues:**
   - When you get an error
   - Copy error message
   - Ask AI: "Fix this error: [paste error]"

---

## 🐛 Common Issues in Windsurf

### Issue 1: Python Interpreter Not Found

**Solution:**
1. Press **Ctrl+Shift+P** (Cmd+Shift+P on Mac)
2. Type: "Python: Select Interpreter"
3. Choose the one in `venv/bin/python`

### Issue 2: Module Not Found

**Solution:**
```bash
# In Windsurf terminal
pip list  # Check installed packages
pip install [missing-package]
```

### Issue 3: Port Already in Use

**Solution:**
```bash
# Find process on port 5000
lsof -ti:5000 | xargs kill -9  # Mac/Linux
# OR
netstat -ano | findstr :5000   # Windows

# Then restart the server
```

### Issue 4: Import Errors

**Solution:**
```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

# Or add to .env file in Windsurf
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Virtual environment activated (see `(venv)` in terminal)
- [ ] All dependencies installed (`pip list` shows packages)
- [ ] Backend server starts without errors
- [ ] Frontend opens in browser
- [ ] Can record audio
- [ ] Translation works
- [ ] Voice cloning applies

---

## 🎯 Next Steps

Once working:

1. **Add Features:**
   - More languages
   - Batch processing
   - Speaker selection

2. **Optimize:**
   - Use GPU if available
   - Cache models
   - Reduce latency

3. **Deploy:**
   - Railway.app
   - Render.com
   - AWS/GCP

4. **Monitor:**
   - Add logging
   - Track performance
   - User analytics

---

## 💡 Windsurf Pro Tips

1. **Use Command Palette:**
   - Press **Ctrl+Shift+P**
   - Access all features

2. **Quick File Navigation:**
   - Press **Ctrl+P**
   - Type filename
   - Jump instantly

3. **Multi-Cursor Editing:**
   - **Alt+Click** to add cursors
   - Edit multiple lines at once

4. **Zen Mode:**
   - Press **Ctrl+K Z**
   - Distraction-free coding

5. **Extensions:**
   - Python
   - Pylance
   - GitLens
   - Better Comments

---

## 🆘 Getting Help

1. **In Windsurf:**
   - Press **Ctrl+L** → Ask AI
   
2. **Documentation:**
   - Read `WINDSURF_INTEGRATION_GUIDE.md`
   
3. **Community:**
   - Windsurf Discord
   - GitHub Issues

---

## 🎉 Success!

You now have a fully integrated speech translation system with voice cloning running in Windsurf!

**What you've achieved:**
✅ Real-time speech translation
✅ Voice characteristic preservation  
✅ Professional development environment
✅ Ready for deployment

**Happy coding in Windsurf! 🌊**
