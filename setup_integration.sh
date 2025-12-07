#!/bin/bash

# 🚀 Quick Integration Setup Script
# This script sets up the integrated CapLine_TazzoX + Voice Cloning project

echo "=================================================="
echo "🚀 Integrated Speech Translation Setup"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"
echo ""

# Create project structure
echo "📁 Creating project structure..."
mkdir -p backend/{capline,voice_cloning,integration}
mkdir -p frontend
mkdir -p models/{whisper,m2m100,akf_models}
mkdir -p outputs
mkdir -p tests/sample_audio

echo "✅ Directories created"
echo ""

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"
echo ""

# Create requirements file
echo "📝 Creating requirements.txt..."
cat > backend/requirements_integrated.txt << 'EOF'
# CapLine TazzoX Dependencies
flask==3.0.0
flask-cors==4.0.0
openai-whisper==20231117
transformers==4.35.0
torch==2.1.0
torchaudio==2.1.0
gtts==2.4.0
pydub==0.25.1

# Voice Cloning Dependencies
librosa==0.10.1
matplotlib==3.8.2
numpy==1.24.3
gradio==4.8.0
soundfile==0.12.1
jiwer==3.0.3
scikit-learn==1.3.2
pandas==2.1.3

# Integration-specific
scipy==1.11.4
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
python-multipart==0.0.6
pyyaml==6.0.1
EOF

echo "✅ Requirements file created"
echo ""

# Install dependencies
echo "📦 Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r backend/requirements_integrated.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""

# Create __init__.py files
echo "📄 Creating module files..."
touch backend/__init__.py
touch backend/capline/__init__.py
touch backend/voice_cloning/__init__.py
touch backend/integration/__init__.py

cat > backend/integration/__init__.py << 'EOF'
"""
Integration layer for CapLine_TazzoX and Voice Cloning
"""

from .integrated_pipeline import IntegratedTranslationPipeline
from .feature_bridge import FeatureBridge

__all__ = ['IntegratedTranslationPipeline', 'FeatureBridge']
EOF

echo "✅ Module files created"
echo ""

# Create a simple test script
echo "🧪 Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Quick setup test script
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('whisper', 'OpenAI Whisper'),
        ('transformers', 'Transformers'),
        ('librosa', 'Librosa'),
        ('flask', 'Flask'),
        ('gradio', 'Gradio'),
    ]
    
    failed = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT FOUND")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r backend/requirements_integrated.txt")
        sys.exit(1)
    else:
        print("\n✅ All packages installed correctly!")

if __name__ == "__main__":
    test_imports()
EOF

chmod +x test_setup.py

echo "✅ Test script created"
echo ""

# Run test
echo "🧪 Testing installation..."
python3 test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "1. Copy your CapLine_TazzoX files to: backend/capline/"
    echo "2. Copy your Voice Cloning files to: backend/voice_cloning/"
    echo "3. Follow the integration guide in WINDSURF_INTEGRATION_GUIDE.md"
    echo ""
    echo "To activate the environment later:"
    echo "  source venv/bin/activate  (Mac/Linux)"
    echo "  venv\\Scripts\\activate    (Windows)"
    echo ""
else
    echo ""
    echo "❌ Setup encountered errors. Please check the output above."
fi
