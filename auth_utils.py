import os
import json
import hashlib

USERS_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        # Seed default demo accounts
        default_users = {
            "admin": {
                "name": "Admin User",
                "password_hash": hash_password("admin123"),
                "role": "Lead ML Engineer",
                "email": "admin@lingai.org"
            },
            "demo": {
                "name": "Speech Researcher",
                "password_hash": hash_password("demo123"),
                "role": "Linguist & Audio Analyst",
                "email": "researcher@lingai.org"
            }
        }
        with open(USERS_FILE, "w") as f:
            json.dump(default_users, f, indent=4)
        return default_users
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def authenticate(username, password):
    users = load_users()
    u = username.strip().lower()
    if u in users:
        if users[u]["password_hash"] == hash_password(password):
            return True, users[u]
    return False, None

def register_user(username, name, email, password, role="Linguistics Researcher"):
    users = load_users()
    u = username.strip().lower()
    if not u or not password:
        return False, "Username and password cannot be empty."
    if u in users:
        return False, f"Username '{username}' is already registered."
    
    users[u] = {
        "name": name.strip() if name else username,
        "password_hash": hash_password(password),
        "role": role,
        "email": email.strip() if email else f"{u}@lingai.org"
    }
    save_users(users)
    return True, "Account created successfully! You can now log in."
