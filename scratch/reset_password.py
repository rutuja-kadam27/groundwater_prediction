# scratch/reset_password.py
import sqlite3
import os
from werkzeug.security import generate_password_hash

db_path = "users.db"

if not os.path.exists(db_path):
    print(f"Error: {db_path} not found.")
    exit(1)

username = input("Enter the username you want to reset: ").strip()
new_password = input("Enter the new plain-text password: ").strip()

if not username or not new_password:
    print("Username and password cannot be empty.")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    if not user:
        print(f"Error: User '{username}' does not exist.")
    else:
        # Generate secure hash
        p_hash = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (p_hash, username))
        conn.commit()
        print(f"Success! Password for user '{username}' has been reset to: '{new_password}'")
except sqlite3.Error as e:
    print("Database error:", e)
finally:
    conn.close()
