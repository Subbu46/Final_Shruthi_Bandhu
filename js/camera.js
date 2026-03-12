// Function to start camera with loading state
function startCamera(videoElementId) {
    const video = document.getElementById(videoElementId);
    const container = document.querySelector('.camera-container');
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    container.appendChild(spinner); // Show loading spinner

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
                container.removeChild(spinner); // Hide spinner
                // Tooltip: Add a small help text
                const tooltip = document.createElement('p');
                tooltip.textContent = 'Perform gestures in front of the camera.';
                tooltip.style.fontSize = '14px';
                tooltip.style.color = '#666';
                container.appendChild(tooltip);
            })
            .catch(function(err) {
                container.removeChild(spinner);
                console.error("Error accessing camera: ", err);
                alert("Camera access denied. Please allow permissions and try again. Error: " + err.message);
            });
    } else {
        container.removeChild(spinner);
        alert("Camera not supported in this browser. Please use Chrome or Firefox.");
    }
}

// Enhanced gesture processing with loading
function processGestureToText() {
    const output = document.getElementById('text-output');
    const button = event.target;
    button.disabled = true;
    button.textContent = 'Processing...';
    output.innerHTML = '<div class="spinner"></div>'; // Show spinner in output

    // Simulate processing delay (replace with real model call)
    setTimeout(() => {
        const recognizedText = "Hello, this is a gesture output!";
        output.textContent = recognizedText;
        speakText(recognizedText);
        button.disabled = false;
        button.textContent = 'Process Gesture';
    }, 2000); // 2-second delay for demo
}