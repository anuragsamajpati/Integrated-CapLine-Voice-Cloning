# CapLine TazzoX - Real-Time Speech Translation

Complete full-stack application with backend API and frontend web interface.

## 🚀 Features

- **Real-time speech translation** to 19 languages
- **Beautiful web interface** with microphone recording
- **REST API backend** using Flask
- **AI-powered** with Whisper ASR and M2M100 translation
- **Text-to-Speech** output in target language

## 📁 Project Structure

```
capline_project/
├── backend/
│   ├── app.py              # Flask API server
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Web interface
│   └── app.js              # Frontend JavaScript
└── README.md               # This file
```

## 🛠️ Setup Instructions

### Backend Setup

1. **Navigate to backend folder:**
   ```bash
   cd backend
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   
   # Activate on Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server:**
   ```bash
   python app.py
   ```

   Server will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend folder:**
   ```bash
   cd frontend
   ```

2. **Option 1: Use VS Code Live Server**
   - Install "Live Server" extension in VS Code
   - Right-click on `index.html`
   - Select "Open with Live Server"

3. **Option 2: Use Python HTTP Server**
   ```bash
   python -m http.server 8000
   ```
   Then open `http://localhost:8000` in your browser

4. **Option 3: Just open the file**
   - Double-click `index.html`
   - Will open in your default browser

## 🎯 How to Use

1. **Start the backend** first (port 5000)
2. **Open the frontend** (port 8000 or any browser)
3. **Select target language** from dropdown
4. **Click "Start Recording"** button (or press SPACE)
5. **Speak in English** for a few seconds
6. **Click "Stop Recording"** (or press SPACE again)
7. **Wait** for processing (~3-5 seconds)
8. **See translation** and **hear the audio**!

## 🌍 Supported Languages

- Hindi
- Spanish
- French
- German
- Japanese
- Portuguese
- Russian
- Arabic
- Turkish
- Chinese
- Bengali
- Telugu
- Marathi
- Tamil
- Gujarati
- Kannada
- Urdu
- Malay
- Indonesian

## 🔧 Configuration

### Change Backend URL

If you deploy the backend to a different server, update the API URL in `frontend/app.js`:

```javascript
const API_URL = 'http://your-backend-url.com';  // Change this
```

### Change Port

Backend port can be changed in `backend/app.py`:

```python
app.run(host='0.0.0.0', port=5000, debug=True)  # Change port here
```

## 📝 API Endpoints

### GET /
Health check

**Response:**
```json
{
  "status": "online",
  "service": "CapLine TazzoX API",
  "version": "1.0"
}
```

### GET /languages
Get list of supported languages

**Response:**
```json
{
  "languages": ["Hindi", "Spanish", "French", ...]
}
```

### POST /translate
Translate audio

**Request:**
- `audio` (file): Audio file (WebM format)
- `target_language` (form): Target language name

**Response:**
```json
{
  "success": true,
  "input_text": "Hello, how are you?",
  "input_language": "en",
  "translated_text": "नमस्ते, आप कैसे हैं?",
  "target_language": "Hindi",
  "audio_base64": "base64_encoded_audio_data"
}
```

## 🚀 Deployment Options

### Option 1: Heroku (Free tier)
```bash
# In backend folder
heroku create your-app-name
git push heroku main
```

### Option 2: Railway.app
- Connect your GitHub repo
- Deploy automatically

### Option 3: Render.com
- Free tier available
- Connect GitHub and deploy

### Option 4: DigitalOcean App Platform
- $5/month
- Easy deployment

### Frontend Deployment:
- **Netlify** (free): Drag & drop frontend folder
- **Vercel** (free): Connect GitHub repo
- **GitHub Pages** (free): Push to gh-pages branch

## ⚙️ System Requirements

**Minimum:**
- Python 3.8+
- 4 GB RAM
- 2 GB free disk space

**Recommended:**
- Python 3.9+
- 8 GB RAM
- 5 GB free disk space
- GPU (optional, for faster processing)

## 🐛 Troubleshooting

### Backend won't start:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is already in use
- Try running with `python3` instead of `python`

### Frontend can't connect to backend:
- Make sure backend is running on port 5000
- Check CORS settings in `app.py`
- Update API_URL in `app.js` if needed

### Microphone not working:
- Grant microphone permission in browser
- Use HTTPS or localhost (required for mic access)
- Check if microphone is working in other apps

### Translation is slow:
- First translation is always slower (model loading)
- Use GPU if available
- Consider using "tiny" Whisper model for speed

## 📦 Building for Production

### Backend:
```bash
# Use production WSGI server
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:5000
```

### Frontend:
- Minify JavaScript and CSS
- Update API_URL to production backend
- Enable HTTPS

## 📄 License

This project is for educational and research purposes.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the setup instructions
3. Ensure all dependencies are installed

## 🎉 Credits

- **Whisper ASR**: OpenAI
- **M2M100 Translation**: Facebook AI
- **gTTS**: Google Text-to-Speech

---

**Built with ❤️ for real-time speech translation**
