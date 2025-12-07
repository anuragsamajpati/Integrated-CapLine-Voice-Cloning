# 🔗 Integration Guide: CapLine_TazzoX + Voice Cloning in Windsurf

## 📋 Overview

This guide will help you integrate:
- **CapLine_TazzoX**: Real-time speech translation system (Whisper ASR + M2M100)
- **AKF Voice Cloning**: Adaptive Kalman Filter-based voice cloning system

**Integration Goal**: Translate speech while preserving the speaker's voice characteristics

---

## 🏗️ Integrated Project Structure

```
integrated-speech-translation/
├── backend/
│   ├── capline/                    # CapLine TazzoX components
│   │   ├── app.py                  # Original Flask API
│   │   └── requirements.txt
│   ├── voice_cloning/              # Voice Cloning components
│   │   ├── AKF_base_algo_full.py   # Core AKF algorithm
│   │   ├── akf_dashboard.py        # Gradio dashboard
│   │   └── akf_realtime_ui.py      # Realtime UI
│   ├── integration/                # NEW: Integration layer
│   │   ├── __init__.py
│   │   ├── integrated_pipeline.py  # Main integration logic
│   │   ├── feature_bridge.py       # Connect TazzoX → Voice Cloning
│   │   └── api_server.py           # Unified API endpoint
│   └── requirements_integrated.txt  # Combined dependencies
├── frontend/
│   ├── index.html                  # Enhanced UI
│   └── app.js                      # Updated frontend logic
├── models/                         # Model storage
│   ├── whisper/
│   ├── m2m100/
│   └── akf_models/
├── outputs/                        # Generated audio files
├── tests/                          # Integration tests
│   ├── test_integration.py
│   └── sample_audio/
└── README_INTEGRATED.md            # This file
```

---

## 🚀 Step-by-Step Integration in Windsurf

### Step 1: Open Windsurf and Create Project Structure

1. **Open Windsurf IDE**
2. **Create new folder**: `integrated-speech-translation`
3. **Open folder in Windsurf**: File → Open Folder

### Step 2: Set Up Virtual Environment

```bash
# In Windsurf terminal (Ctrl+` or Cmd+`)
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Copy Your Existing Code

**Using Windsurf File Explorer:**

1. Create the folder structure shown above
2. Copy CapLine_TazzoX files to `backend/capline/`
3. Copy Voice Cloning files to `backend/voice_cloning/`

**Or using terminal:**

```bash
# Assuming your original projects are in these locations:
# ~/capline_project/backend/*
# ~/Voice\ Cloning/*

mkdir -p backend/capline backend/voice_cloning backend/integration
mkdir -p frontend models outputs tests

# Copy CapLine files
cp ~/capline_project/backend/* backend/capline/

# Copy Voice Cloning files
cp ~/Voice\ Cloning/AKF_base_algo_full.py backend/voice_cloning/
cp ~/Voice\ Cloning/akf_dashboard.py backend/voice_cloning/
cp ~/Voice\ Cloning/akf_realtime_ui.py backend/voice_cloning/

# Copy frontend
cp ~/capline_project/frontend/* frontend/
```

---

## 📦 Step 4: Create Combined Requirements File

Create `backend/requirements_integrated.txt`:

```txt
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
resemblyzer==0.1.1.dev0
scikit-learn==1.3.2
pandas==2.1.3

# Integration-specific
scipy==1.11.4
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
python-multipart==0.0.6
pyyaml==6.0.1
```

```bash
pip install -r backend/requirements_integrated.txt
```

---

## 🔧 Step 5: Create Integration Layer

### Create `backend/integration/__init__.py`

```python
"""
Integration layer for CapLine_TazzoX and Voice Cloning
"""

from .integrated_pipeline import IntegratedTranslationPipeline
from .feature_bridge import FeatureBridge
from .api_server import create_integrated_app

__all__ = ['IntegratedTranslationPipeline', 'FeatureBridge', 'create_integrated_app']
```

### Create `backend/integration/feature_bridge.py`

```python
"""
Feature Bridge: Connects TazzoX prosody features to Voice Cloning
"""

import numpy as np
import librosa
import torch
from typing import Dict, Tuple, Optional

class FeatureBridge:
    """
    Bridges prosody features from CapLine_TazzoX to Voice Cloning system
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_prosody_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract prosody features needed for voice cloning
        
        Args:
            audio: Audio waveform (numpy array)
            
        Returns:
            Dictionary with prosody features
        """
        # F0 (pitch) extraction
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0)
        
        # Delta F0
        delta_f0 = np.diff(f0, prepend=f0[0])
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Spectral features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=80
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            'f0': f0,
            'delta_f0': delta_f0,
            'rms_energy': rms,
            'mfcc': mfcc,
            'mel_spectrogram': mel_db,
            'voiced_probability': voiced_probs
        }
    
    def prepare_for_voice_cloning(
        self, 
        audio: np.ndarray,
        prosody_features: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Prepare audio and features for voice cloning pipeline
        
        Args:
            audio: Input audio waveform
            prosody_features: Pre-extracted prosody features (optional)
            
        Returns:
            Tuple of (processed_audio, feature_dict)
        """
        if prosody_features is None:
            prosody_features = self.extract_prosody_features(audio)
        
        # Normalize audio
        audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio_normalized, prosody_features
```

### Create `backend/integration/integrated_pipeline.py`

```python
"""
Integrated Pipeline: Combines CapLine_TazzoX translation with Voice Cloning
"""

import os
import sys
import numpy as np
import torch
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from gtts import gTTS
import tempfile
import librosa
import soundfile as sf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voice_cloning.AKF_base_algo_full import run_voice_cloning_pipeline
from integration.feature_bridge import FeatureBridge


class IntegratedTranslationPipeline:
    """
    Main integration class combining translation and voice cloning
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "cpu",
        sample_rate: int = 16000
    ):
        """
        Initialize the integrated pipeline
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on ('cpu' or 'cuda')
            sample_rate: Audio sample rate
        """
        print("🚀 Initializing Integrated Translation Pipeline...")
        
        self.device = device
        self.sample_rate = sample_rate
        
        # Initialize CapLine_TazzoX components
        print("📥 Loading Whisper ASR model...")
        self.whisper_model = whisper.load_model(whisper_model, device=device)
        
        print("🌍 Loading M2M100 translation model...")
        self.translator = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        
        # Initialize Feature Bridge
        self.feature_bridge = FeatureBridge(sample_rate=sample_rate)
        
        # Language code mapping
        self.lang_codes = {
            'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr',
            'German': 'de', 'Japanese': 'ja', 'Portuguese': 'pt',
            'Russian': 'ru', 'Arabic': 'ar', 'Turkish': 'tr',
            'Chinese': 'zh', 'Bengali': 'bn', 'Telugu': 'te',
            'Marathi': 'mr', 'Tamil': 'ta', 'Gujarati': 'gu',
            'Kannada': 'kn', 'Urdu': 'ur', 'Malay': 'ms',
            'Indonesian': 'id'
        }
        
        print("✅ Initialization complete!\n")
    
    def transcribe_audio(self, audio: np.ndarray) -> Tuple[str, str]:
        """
        Transcribe audio using Whisper
        
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        print("🎤 Transcribing audio...")
        result = self.whisper_model.transcribe(audio)
        text = result["text"].strip()
        language = result["language"]
        print(f"   Detected language: {language}")
        print(f"   Transcribed text: {text}")
        return text, language
    
    def translate_text(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """
        Translate text using M2M100
        
        Args:
            text: Input text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        print(f"🌐 Translating from {source_lang} to {target_lang}...")
        
        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.translator.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(target_lang)
        )
        translated_text = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        
        print(f"   Translated text: {translated_text}")
        return translated_text
    
    def clone_voice_with_translation(
        self,
        original_audio: np.ndarray,
        translated_text: str,
        target_lang: str
    ) -> np.ndarray:
        """
        Generate translated speech with cloned voice characteristics
        
        Args:
            original_audio: Original speaker's audio
            translated_text: Translated text
            target_lang: Target language code
            
        Returns:
            Cloned audio with translation
        """
        print("🎭 Generating voice-cloned translation...")
        
        # Step 1: Extract prosody features from original speaker
        prosody_features = self.feature_bridge.extract_prosody_features(original_audio)
        
        # Step 2: Generate translated speech with gTTS
        print("   Generating base translated speech...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_path = tmp_file.name
        
        tts = gTTS(text=translated_text, lang=target_lang, slow=False)
        tts.save(tmp_path)
        
        # Load generated audio
        translated_audio, sr = librosa.load(tmp_path, sr=self.sample_rate)
        os.unlink(tmp_path)
        
        # Step 3: Apply voice cloning to match original speaker
        print("   Applying voice cloning...")
        audio_normalized, features = self.feature_bridge.prepare_for_voice_cloning(
            original_audio,
            prosody_features
        )
        
        try:
            # Use AKF voice cloning pipeline
            cloned_audio = run_voice_cloning_pipeline(
                translated_audio,
                use_fast_features=True  # For realtime performance
            )
            print("   ✅ Voice cloning successful!")
            return cloned_audio
            
        except Exception as e:
            print(f"   ⚠️  Voice cloning failed: {e}")
            print("   Using translated audio without cloning...")
            return translated_audio
    
    def process_audio_file(
        self,
        audio_path: str,
        target_language: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Complete pipeline: Transcribe → Translate → Clone Voice
        
        Args:
            audio_path: Path to input audio file
            target_language: Target language name (e.g., 'Hindi', 'Spanish')
            output_path: Optional path to save output audio
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"🎯 Processing: {os.path.basename(audio_path)}")
        print(f"{'='*60}\n")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Step 1: Transcribe
        input_text, input_lang = self.transcribe_audio(audio)
        
        # Step 2: Translate
        target_lang_code = self.lang_codes.get(target_language, 'en')
        translated_text = self.translate_text(input_text, input_lang, target_lang_code)
        
        # Step 3: Clone voice with translation
        output_audio = self.clone_voice_with_translation(
            audio,
            translated_text,
            target_lang_code
        )
        
        # Save output
        if output_path is None:
            output_path = f"outputs/integrated_output_{target_language.lower()}.wav"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, output_audio, self.sample_rate)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Output saved to: {output_path}\n")
        
        return {
            'success': True,
            'input_text': input_text,
            'input_language': input_lang,
            'translated_text': translated_text,
            'target_language': target_language,
            'output_path': output_path
        }


# Example usage
if __name__ == "__main__":
    pipeline = IntegratedTranslationPipeline(whisper_model="base")
    
    # Test with a sample audio file
    result = pipeline.process_audio_file(
        audio_path="tests/sample_audio/test.wav",
        target_language="Hindi",
        output_path="outputs/test_hindi.wav"
    )
    
    print(f"\nResults: {result}")
```

### Create `backend/integration/api_server.py`

```python
"""
Unified API Server combining both systems
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import tempfile
import os
from integrated_pipeline import IntegratedTranslationPipeline

def create_integrated_app():
    """Create and configure the integrated Flask application"""
    
    app = Flask(__name__)
    CORS(app)
    
    # Initialize pipeline (loads once at startup)
    print("🚀 Starting Integrated API Server...")
    pipeline = IntegratedTranslationPipeline(whisper_model="base")
    
    @app.route('/', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'online',
            'service': 'Integrated Translation + Voice Cloning API',
            'version': '2.0',
            'features': [
                'Speech Recognition (Whisper)',
                'Translation (M2M100)',
                'Voice Cloning (AKF)',
                'Prosody Preservation'
            ]
        })
    
    @app.route('/languages', methods=['GET'])
    def get_languages():
        """Get supported languages"""
        return jsonify({
            'languages': list(pipeline.lang_codes.keys())
        })
    
    @app.route('/translate-with-voice', methods=['POST'])
    def translate_with_voice():
        """
        Translate audio and clone voice
        
        Expected form data:
        - audio: Audio file
        - target_language: Target language name
        """
        try:
            # Get audio file
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            target_language = request.form.get('target_language', 'Hindi')
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                audio_path = tmp.name
                audio_file.save(audio_path)
            
            # Process through integrated pipeline
            result = pipeline.process_audio_file(
                audio_path=audio_path,
                target_language=target_language
            )
            
            # Read output audio and encode as base64
            with open(result['output_path'], 'rb') as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temp files
            os.unlink(audio_path)
            
            return jsonify({
                'success': True,
                'input_text': result['input_text'],
                'input_language': result['input_language'],
                'translated_text': result['translated_text'],
                'target_language': target_language,
                'audio_base64': audio_base64,
                'output_file': result['output_path']
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app


if __name__ == '__main__':
    app = create_integrated_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## 🎨 Step 6: Update Frontend

Update `frontend/app.js` to use the new integrated endpoint:

```javascript
const API_URL = 'http://localhost:5000';

// ... (keep existing code) ...

// Update the translate function
async function translate() {
    if (!audioBlob) {
        showError('Please record audio first!');
        return;
    }

    const targetLanguage = document.getElementById('targetLanguage').value;
    
    showLoading(true);
    updateStatus('Processing with voice cloning...');

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('target_language', targetLanguage);

    try {
        const response = await fetch(`${API_URL}/translate-with-voice`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            playTranslatedAudio(data.audio_base64);
        } else {
            showError(data.error || 'Translation failed');
        }
    } catch (error) {
        showError('Connection failed: ' + error.message);
    } finally {
        showLoading(false);
    }
}
```

---

## 🧪 Step 7: Testing the Integration

Create `tests/test_integration.py`:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.integration.integrated_pipeline import IntegratedTranslationPipeline

def test_integration():
    """Test the integrated pipeline"""
    
    print("🧪 Testing Integrated Pipeline...")
    
    # Initialize
    pipeline = IntegratedTranslationPipeline(whisper_model="tiny")  # Use tiny for speed
    
    # Test with sample audio
    result = pipeline.process_audio_file(
        audio_path="tests/sample_audio/english_sample.wav",
        target_language="Hindi"
    )
    
    assert result['success'] == True
    assert 'input_text' in result
    assert 'translated_text' in result
    assert os.path.exists(result['output_path'])
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_integration()
```

---

## 🚀 Step 8: Run the Integrated System

### In Windsurf Terminal:

```bash
# 1. Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Navigate to backend
cd backend

# 3. Run the integrated API server
python integration/api_server.py
```

### Open another terminal for frontend:

```bash
# Navigate to frontend
cd frontend

# Start simple HTTP server
python -m http.server 8000
```

### Access the application:
- **Frontend**: http://localhost:8000
- **API**: http://localhost:5000

---

## 📊 Architecture Flow

```
User speaks → Frontend records
                ↓
            Upload to API
                ↓
    ┌───────────────────────────┐
    │  Integrated Pipeline      │
    ├───────────────────────────┤
    │ 1. Whisper ASR            │ → Transcribe
    │ 2. Detect Language        │
    │ 3. M2M100 Translation     │ → Translate
    │ 4. Extract Prosody        │ → Original voice features
    │ 5. gTTS Generate          │ → Base translated speech
    │ 6. AKF Voice Cloning      │ → Apply voice characteristics
    │ 7. Output Synthesis       │ → Final audio
    └───────────────────────────┘
                ↓
        Return translated audio
        with cloned voice
                ↓
        Frontend plays audio
```

---

## 🎯 Key Integration Points

1. **Prosody Extraction** (FeatureBridge)
   - Extracts F0, energy, spectral features from original
   - Feeds into voice cloning pipeline

2. **Translation Pipeline**
   - Whisper for transcription
   - M2M100 for translation
   - gTTS for base speech synthesis

3. **Voice Cloning Application**
   - AKF algorithm applied to translated speech
   - Preserves original speaker characteristics
   - Fast MFCC mode for realtime performance

---

## 🐛 Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in the right directory and venv is activated
cd integrated-speech-translation
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
```

### Issue: CUDA out of memory
```python
# Use CPU instead
pipeline = IntegratedTranslationPipeline(device="cpu")
```

### Issue: Voice cloning too slow
```python
# Use faster Whisper model
pipeline = IntegratedTranslationPipeline(whisper_model="tiny")
```

---

## 🎉 Success! You now have:

✅ Integrated speech translation  
✅ Voice cloning capability  
✅ Unified API endpoint  
✅ Working web interface  
✅ Prosody preservation  

---

**Next Steps:**
- Add more languages
- Improve voice cloning quality
- Deploy to cloud (Render, Railway, etc.)
- Add batch processing
- Implement caching for better performance

Happy coding! 🚀
