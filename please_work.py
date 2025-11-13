import sqlite3
import json

conn = sqlite3.connect("users.db")
conn.row_factory = sqlite3.Row  # optional, allows accessing columns by name
cur = conn.cursor()

cur.execute("SELECT * FROM feedback_json")
rows = cur.fetchall()

for row in rows:
    feedback_data = json.loads(row["feedback_json"])  # parse JSON string back to dict
    print(feedback_data)

conn.close()

