import sqlite3

DB_FILE = "users.db"

with sqlite3.connect(DB_FILE) as conn:
    try:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        print("✅ Column 'email' added successfully!")
    except sqlite3.OperationalError as e:
        print("⚠️ Error:", e)
