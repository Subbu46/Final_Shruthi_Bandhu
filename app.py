
import numpy as np
from flask import request, jsonify



from flask import Flask, render_template, Response, jsonify, send_from_directory
from inference_backend import GestureSystem
import os
# Update your imports line to include 'send_from_directory'
# Initialize Flask
# template_folder='.' tells Flask to look for HTML files in the MAIN folder
app = Flask(__name__, template_folder='.', static_folder='.')
system = GestureSystem()

# --- HINDI TRANSLATION MAP ---
TRANSLATION_MAP = {
    # System States
    "Uncertain": "अनिश्चित",
    "Initializing...": "प्रारंभ हो रहा है...",
    "Waiting...": "प्रतीक्षा...",

    # Model Classes
    "Again": "फिर से",
    "Beautiful": "सुंदर",
    "Camera": "कैमरा",
    "Correct": "सही",
    "Food": "खाना",
    "Good Evening": "शुभ संध्या",
    "Good Morning": "शुभ प्रभात",
    "Goodbye": "अलविदा",
    "I": "मैं",
    "Language": "भाषा",
    "Namasthe": "नमस्ते",
    "Short": "छोटा",
    "Sign": "संकेत",
    "Than You Very Much": "आपका बहुत-बहुत धन्यवाद",
    "Water": "पानी",
    "Work": "काम",
    "Wrong": "गलत",
    "You": "आप"
}

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/egocentric')
def egocentric():
    system.connect_camera()
    return render_template('egocentric.html')

# --- NEW: ROUTE FOR ALL OTHER PAGES (Login, Signup, About, etc.) ---
@app.route('/<page_name>.html')
def render_static_pages(page_name):
    """
    This function handles requests for 'login.html', 'about.html', etc.
    It automatically finds the matching file and serves it.
    """
    return render_template(f'{page_name}.html')

@app.route('/video_feed')
def video_feed():
    if not system.camera:
        return "", 204
    return Response(
        system.generate_jpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_prediction')
def get_prediction():
    english_pred = system.latest_prediction
    hindi_pred = TRANSLATION_MAP.get(english_pred, english_pred)
    
    return jsonify({
        'english': english_pred,
        'hindi': hindi_pred,
        'confidence': system.latest_confidence
    })


# --- STATIC FILE ROUTES ---
# These tell Flask how to serve your CSS, JS, and Images

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)


#Exocentric api endpoint
@app.route("/api/exocentric/infer", methods=["POST"])
def exocentric_infer():
    """
    Expects JSON:
    {
      "sequence": [
        [f1, f2, ..., fD],
        [f1, f2, ..., fD],
        ...
      ]
    }
    """

    data = request.get_json()
    if data is None or "sequence" not in data:
        return jsonify({"error": "Invalid payload"}), 400

    try:
        seq = np.array(data["sequence"], dtype=np.float32)

        if seq.ndim != 2:
            return jsonify({"error": "Invalid sequence shape"}), 400

        result = infer_exocentric(seq)

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        print("[EXO ERROR]", e)
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    # Running on port 8080 as per your screenshot
    app.run(host='0.0.0.0', port=8080, debug=False)
