import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_FILE = "users.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Users table with email
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # History table
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            file TEXT,
            time TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def add_user(username, email, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hashed_pw = generate_password_hash(password)
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                  (username, email, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username or email already exists
        return False
    finally:
        conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_email(email):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def verify_user(username_or_email, password):
    user = get_user_by_username(username_or_email) or get_user_by_email(username_or_email)
    if user and check_password_hash(user[3], password):
        return user
    return None

def add_history(user_id, action, filename, time):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO history (user_id, action, file, time) VALUES (?, ?, ?, ?)",
              (user_id, action, filename, time))
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT action, file, time FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 5", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"action": r[0], "file": r[1], "time": r[2]} for r in rows]

# Initialize the database when this file is imported
init_db()
