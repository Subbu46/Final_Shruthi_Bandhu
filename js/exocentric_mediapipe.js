let holistic;
let camera;
let recording = false;
let sequence = [];
let silenceStart = null;

const HAND_HEIGHT_THRESHOLD = 0.9;
const SILENCE_THRESHOLD = 1000; // ms
const MIN_FRAMES = 15;

const video = document.getElementById("exoVideo");

function getWristY(results) {
    let ys = [];
    if (results.poseLandmarks) {
        ys.push(results.poseLandmarks[15].y);
        ys.push(results.poseLandmarks[16].y);
    }
    return ys.length ? (ys[0] + ys[1]) / 2 : 1.0;
}

function extractFrame(results) {
    let pose = results.poseLandmarks
        ? results.poseLandmarks.flatMap(p => [p.x, p.y])
        : new Array(66).fill(0);

    function handOrWrist(hand, wristIdx) {
        if (hand) return hand.landmarks.flatMap(p => [p.x, p.y]);
        if (results.poseLandmarks) {
            const w = results.poseLandmarks[wristIdx];
            return Array(21).fill([w.x, w.y]).flat();
        }
        return new Array(42).fill(0);
    }

    const lh = handOrWrist(results.leftHandLandmarks, 15);
    const rh = handOrWrist(results.rightHandLandmarks, 16);

    return [...pose, ...lh, ...rh];
}

async function onResults(results) {
    const wristY = getWristY(results);

    if (wristY < HAND_HEIGHT_THRESHOLD) {
        if (!recording) {
            recording = true;
            sequence = [];
            silenceStart = null;
            setUIState("recording");
        }
        sequence.push(extractFrame(results));
    } else if (recording) {
        if (!silenceStart) silenceStart = Date.now();
        if (Date.now() - silenceStart > SILENCE_THRESHOLD) {
            recording = false;
            setUIState("processing");

            if (sequence.length >= MIN_FRAMES) {
                sendSequence(sequence);
            } else {
                setUIState("waiting");
            }
        }
    }
}

holistic = new Holistic({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
});

holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

holistic.onResults(onResults);

camera = new Camera(video, {
    onFrame: async () => {
        await holistic.send({ image: video });
    },
    width: 640,
    height: 480
});

camera.start();
