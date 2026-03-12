// Firebase config - replace with your values from Firebase Console
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyDgzuxY0qc3pienSsEC-q2vB70Qkmml-A8",
  authDomain: "shruthi-bandhu-9846c.firebaseapp.com",
  projectId: "shruthi-bandhu-9846c",
  storageBucket: "shruthi-bandhu-9846c.firebasestorage.app",
  messagingSenderId: "413692146435",
  appId: "1:413692146435:web:7f5c2c2643c05d75c6af36"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);