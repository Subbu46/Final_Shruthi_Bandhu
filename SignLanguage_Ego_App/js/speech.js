if (typeof currentLang === "undefined") {
  var currentLang = "en";
}

if (typeof translations === "undefined") {
  var translations = {};
}


let audioEnabled = true;

function toggleAudio() {
    audioEnabled = !audioEnabled;
    const button = document.getElementById('audio-toggle');
    button.textContent = audioEnabled ? translations[currentLang].disableAudio : translations[currentLang].enableAudio;
}

function speakText(text) {
    if (audioEnabled && 'speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = currentLang === 'hi' ? 'hi-IN' : 'en-US';  // Set language for speech
        window.speechSynthesis.speak(utterance);
    } else if (!audioEnabled) {
        console.log('Audio disabled: Text not spoken.');
    } else {
        alert("Text-to-speech not supported.");
    }
}