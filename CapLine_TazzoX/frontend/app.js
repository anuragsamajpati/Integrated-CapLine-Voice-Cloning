// CapLine TazzoX - Frontend JavaScript
// Handles audio recording and API communication

const API_URL = 'http://localhost:5000'; // Change this to your deployed backend URL

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Get DOM elements
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const languageSelect = document.getElementById('languageSelect');
const status = document.getElementById('status');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const inputText = document.getElementById('inputText');
const translatedText = document.getElementById('translatedText');
const audioPlayer = document.getElementById('audioPlayer');

// Show status message
function showStatus(message, type = 'info') {
    status.textContent = message;
    status.className = `status ${type}`;
    status.style.display = 'block';
}

// Hide status
function hideStatus() {
    status.style.display = 'none';
}

// Show loader
function showLoader() {
    loader.style.display = 'block';
}

// Hide loader
function hideLoader() {
    loader.style.display = 'none';
}

// Start recording
async function startRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Create media recorder
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        // Collect audio data
        mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
        });
        
        // Handle recording stop
        mediaRecorder.addEventListener('stop', () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            sendAudioToBackend(audioBlob);
        });
        
        // Start recording
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        recordBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        showStatus('🎤 Recording... Speak now!', 'recording');
        results.style.display = 'none';
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showStatus('❌ Error: Could not access microphone. Please grant permission.', 'error');
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Stop all audio tracks
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        // Update UI
        recordBtn.style.display = 'block';
        stopBtn.style.display = 'none';
        showStatus('⏳ Processing your audio...', 'info');
        showLoader();
    }
}

// Send audio to backend
async function sendAudioToBackend(audioBlob) {
    const targetLanguage = languageSelect.value;
    
    // Create form data
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('target_language', targetLanguage);
    
    try {
        // Send to backend
        const response = await fetch(`${API_URL}/translate`, {
            method: 'POST',
            body: formData
        });
        
        hideLoader();
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Display results
            inputText.textContent = data.input_text;
            translatedText.textContent = data.translated_text;
            
            // Set audio player
            const audioData = `data:audio/mp3;base64,${data.audio_base64}`;
            audioPlayer.src = audioData;
            
            // Show results
            results.style.display = 'block';
            showStatus('✅ Translation complete!', 'success');
            
            // Auto-play audio
            setTimeout(() => {
                audioPlayer.play().catch(e => console.log('Auto-play prevented:', e));
            }, 500);
            
        } else {
            showStatus('❌ Translation failed. Please try again.', 'error');
        }
        
    } catch (error) {
        hideLoader();
        console.error('Error:', error);
        showStatus(`❌ Error: ${error.message}. Make sure the backend server is running.`, 'error');
    }
}

// Event listeners
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Check if backend is running on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/`);
        if (response.ok) {
            const data = await response.json();
            console.log('Backend connected:', data);
        }
    } catch (error) {
        showStatus('⚠️ Warning: Backend server not connected. Start the backend first.', 'error');
    }
});

// Keyboard shortcut: Space to start/stop recording
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !isRecording && e.target.tagName !== 'SELECT') {
        e.preventDefault();
        startRecording();
    } else if (e.code === 'Space' && isRecording) {
        e.preventDefault();
        stopRecording();
    }
});

console.log('CapLine TazzoX initialized!');
console.log('Tip: Press SPACE to start/stop recording');
