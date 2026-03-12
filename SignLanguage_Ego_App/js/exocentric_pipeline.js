let audioEnabled = true;

function setUIState(state, text = "") {
    const box = document.getElementById("outputBox");
    box.className = "output " + state;

    if (state === "waiting") box.innerText = "Waiting for gesture...";
    if (state === "recording") box.innerText = "Recording...";
    if (state === "processing") box.innerText = "Processing...";
    if (state === "output") box.innerText = text;
}

async function sendSequence(sequence) {
    try {
        const res = await fetch("/exocentric_infer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sequence })
        });

        const data = await res.json();
        setUIState("output", data.sentence);

        if (audioEnabled) speakText(data.sentence);
    } catch (e) {
        setUIState("waiting");
        console.error(e);
    }
}

document.getElementById("audioToggle").onclick = () => {
    audioEnabled = !audioEnabled;
    document.getElementById("audioToggle").innerText =
        audioEnabled ? "Disable Audio" : "Enable Audio";
};
