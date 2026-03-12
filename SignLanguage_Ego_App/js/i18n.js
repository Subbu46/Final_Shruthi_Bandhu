const translations = {
    en: {
        home: "Home",
        about: "About",
        features: "Features",
        learning: "Learning",
        faq: "FAQ",
        contact: "Contact",
        login: "Login",
        signup: "Sign Up",
        egocentric: "Egocentric Conversion",
        exocentric: "Exocentric Conversion",
        processGesture: "Process Gesture",
        disableAudio: "Disable Audio",
        enableAudio: "Enable Audio",
        text: "Text",
        hello: "Hello",
        namaste: "Namaste",
        water: "Water",
        feedback: "Feedback",
        submit: "Submit",
        rating: "Rating",
        comment: "Comment",
        privacy: "Privacy Policy",
        terms: "Terms of Service"
    },
    hi: {
        home: "होम",
        about: "हमारे बारे में",
        features: "विशेषताएँ",
        learning: "सीखना",
        faq: "सामान्य प्रश्न",
        contact: "संपर्क",
        login: "लॉग इन",
        signup: "साइन अप",
        egocentric: "एगोसेन्ट्रिक रूपांतरण",
        exocentric: "एक्सोसेन्ट्रिक रूपांतरण",
        processGesture: "इशारे को संसाधित करें",
        disableAudio: "ऑडियो अक्षम करें",
        enableAudio: "ऑडियो सक्षम करें",
        text: "पाठ",
        hello: "नमस्ते",
        namaste: "नमस्ते",
        water: "पानी",
        feedback: "प्रतिक्रिया",
        submit: "सबमिट करें",
        rating: "रेटिंग",
        comment: "टिप्पणी",
        privacy: "गोपनीयता नीति",
        terms: "सेवा की शर्तें"
    }
};

let currentLang = localStorage.getItem('lang') || 'en';

function setLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('lang', lang);
    updateTexts();
}

function updateTexts() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        el.textContent = translations[currentLang][key] || key;
    });
}

document.addEventListener('DOMContentLoaded', () => {
    updateTexts();
    document.getElementById('lang-selector').addEventListener('change', (e) => {
        setLanguage(e.target.value);
    });
});