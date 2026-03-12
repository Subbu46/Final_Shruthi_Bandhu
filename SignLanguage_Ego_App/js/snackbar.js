function showSnackbar(message, type = "info") {
    const snackbar = document.getElementById("snackbar");
    if (!snackbar) return;

    snackbar.textContent = message;

    // type aesthetics
    if (type === "success") snackbar.style.background = "#22c55e";
    else if (type === "error") snackbar.style.background = "#ef4444";
    else snackbar.style.background = "#3b82f6";

    snackbar.classList.add("show");
    setTimeout(() => snackbar.classList.remove("show"), 3500);
}
