# scratch/view_users.py
import sqlite3
import os

db_path = "users.db"

if not os.path.exists(db_path):
    print(f"Error: {db_path} not found. Make sure the database exists.")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

try:
    cursor.execute("SELECT id, username, email, password_hash, created_at FROM users")
    rows = cursor.fetchall()
    
    if not rows:
        print("No registered users found in the database.")
    else:
        print(f"{'ID':<5} | {'Username':<15} | {'Email':<25} | {'Password Hash (Encrypted)':<40}")
        print("-" * 95)
        for row in rows:
            print(f"{row['id']:<5} | {row['username']:<15} | {row['email']:<25} | {row['password_hash'][:40]}...")
except sqlite3.Error as e:
    print("Database error:", e)
finally:
    conn.close()
