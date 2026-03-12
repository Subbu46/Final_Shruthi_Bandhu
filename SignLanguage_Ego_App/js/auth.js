import { auth, db } from "./firebase-config.js";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  sendPasswordResetEmail,
  onAuthStateChanged,
  signOut,
  updateProfile // <-- Crucial: Allows us to save the display name immediately
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";
import { doc, setDoc, getDoc } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";

const provider = new GoogleAuthProvider();

// ==========================================
// 1. SIGN UP (Handles Username)
// ==========================================
export async function signup(email, password, age, gender, username) {
  console.log("Starting signup for:", email);
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    // A. Update the Auth Profile with the Username immediately
    // This allows the website to welcome the user by name right after signup
    await updateProfile(user, {
        displayName: username
    });

    // B. Save user details to Firestore Database
    await setDoc(doc(db, "users", user.uid), {
      email,
      username: username, // Saved here for database reference
      age: parseInt(age),
      gender,
      provider: "local",
      createdAt: new Date()
    });
    console.log("User saved to Firestore");

    // C. Send welcome email (if EmailJS is configured)
    if (typeof emailjs !== 'undefined') {
      try {
        console.log("Sending welcome email...");
        await emailjs.send("service_ls5vfns", "template_bxqd7uw", { 
            to_email: email, 
            to_name: username 
        });
        console.log("Welcome email sent");
      } catch (emailError) {
        console.warn("Email send failed (non-fatal):", emailError);
      }
    }

    if (typeof showSnackbar === 'function') {
      showSnackbar("Signup successful! Redirecting...", "success");
    } else {
      alert("Signup successful!");
    }
    window.location.href = "index.html";

  } catch (error) {
    console.error("Signup failed:", error);
    if (typeof showSnackbar === 'function') {
      showSnackbar("Signup failed: " + error.message, "error");
    } else {
      alert("Signup failed: " + error.message);
    }
  }
}

// ==========================================
// 2. LOGIN (Forgiving Logic)
// ==========================================
export async function login(email, password) {
  console.log("Starting login for:", email);
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    // Check if user profile exists in Firestore
    try {
        const userDocRef = doc(db, "users", user.uid);
        const userDoc = await getDoc(userDocRef);
        
        // If missing, create it on the fly (Self-healing)
        if (!userDoc.exists()) {
            console.warn("Profile missing. Creating default profile...");
            await setDoc(userDocRef, {
                email: user.email,
                // Try to get name from Auth, or fallback to "User"
                username: user.displayName || "User", 
                provider: "recovered",
                lastLogin: new Date()
            });
        }
    } catch (dbError) {
        console.error("Database check failed, allowing login anyway:", dbError);
    }

    console.log("Login successful");
    if (typeof showSnackbar === 'function') {
      showSnackbar("Login successful! Redirecting...", "success");
    }
    
    window.location.href = "index.html";

  } catch (error) {
    console.error("Login failed:", error);
    let errorMessage = "Login failed: " + error.message;
    
    if (error.code === "auth/user-not-found" || error.code === "auth/wrong-password") {
        errorMessage = "Invalid email or password.";
    }
    
    if (typeof showSnackbar === 'function') {
      showSnackbar(errorMessage, "error");
    } else {
      alert(errorMessage);
    }
  }
}

// ==========================================
// 3. GOOGLE SIGN-IN
// ==========================================
export async function googleSignIn() {
  try {
    const result = await signInWithPopup(auth, provider);
    const user = result.user;
    
    // Ensure profile exists in Firestore
    const userDocRef = doc(db, "users", user.uid);
    const userDoc = await getDoc(userDocRef);
    if (!userDoc.exists()) {
      await setDoc(userDocRef, {
        email: user.email,
        username: user.displayName, // Capture Google Name
        provider: "google",
        createdAt: new Date()
      });
    }
    
    alert("Google login successful!");
    window.location.href = "index.html";
  } catch (error) {
    console.error("Google login error:", error);
    alert("Google login failed: " + error.message);
  }
}

// ==========================================
// 4. FORGOT PASSWORD
// ==========================================
export async function forgotPassword(email) {
  try {
    await sendPasswordResetEmail(auth, email);
    alert("Password reset email sent! Check your inbox.");
  } catch (error) {
    console.error("Reset password error:", error);
    alert("Error: " + error.message);
  }
}

// ==========================================
// 5. AUTH CHECK
// ==========================================
export function checkAuth() {
  return new Promise((resolve, reject) => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      unsubscribe();
      if (user) {
        resolve(user);
      } else {
        reject("Not authenticated");
      }
    });
  });
}

// ==========================================
// 6. LOGOUT
// ==========================================
export async function logout() {
  try {
    await signOut(auth);
    console.log("User logged out");
    window.location.href = "login.html";
  } catch (error) {
    console.error("Logout failed:", error);
    alert("Logout failed: " + error.message);
  }
}