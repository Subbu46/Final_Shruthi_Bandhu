from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera

app = Flask(__name__)
camera = VideoCamera()
latest_prediction_storage = ""

# --- TRANSLATION DICTIONARY ---
# IMPORTANT: Add your NEW gestures here if you want Hindi translation.
# --- TRANSLATION DICTIONARY ---
# Keys are all lowercase to match the normalized input from app.py
TRANSLATION_MAP = {
    "are you free today": "क्या आप आज फ्री हैं?",
    "are you hiding something": "क्या आप कुछ छुपा रहे हैं?",
    "bring water for me": "मेरे लिए पानी लाओ",
    "can i help you": "क्या मैं आपकी मदद कर सकता हूँ?",
    "can you repeat that please": "क्या आप उसे दोहरा सकते हैं?",
    "comb your hair": "अपने बाल कंघी करो",
    "congratulations": "बधाई हो",
    "could you please talk slower": "क्या आप धीरे बात कर सकते हैं?",
    "do me a favour": "मेरा एक काम कर दो",
    "do not abuse him": "उसे गाली मत दो",
    "do not be stubborn": "जिद्दी मत बनो",
    "do not hurt me": "मुझे चोट मत पहुँचाओ",
    "do not make me angry": "मुझे गुस्सा मत दिलाओ",
    "do not take it to the heart": "इसे दिल पर मत लो",
    "do not worry": "चिंता मत करो",
    "do you need something": "क्या आपको कुछ चाहिए?",
    "go and sleep": "जाओ और सो जाओ",
    "he is going into the room": "वह कमरे में जा रहा है",
    "no need to worry dont worry": "चिंता करने की कोई जरूरत नहीं है",
    "this place is beautiful": "यह जगह बहुत सुंदर है"
}
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    global latest_prediction_storage
    while True:
        # Get frame and potential new prediction
        frame, pred = camera.get_frame()
        
        if frame is None: break
        
        # If the camera detected a new gesture, update storage
        if pred: 
            latest_prediction_storage = pred 
            
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global latest_prediction_storage
    
    english_text = latest_prediction_storage
    
    # Reset global after reading so we don't repeat the audio
    latest_prediction_storage = "" 
    
    if english_text:
        # Case insensitive lookup
        lower_key = english_text.lower().strip()
        hindi_text = TRANSLATION_MAP.get(lower_key, english_text) # Fallback to English if no Hindi found

        return jsonify({
            'english': english_text,
            'hindi': hindi_text
        })
    
    return jsonify({'english': '', 'hindi': ''})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
