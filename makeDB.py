'''
建立資料庫與資料表
'''
import sqlite3
conn = sqlite3.connect("linebot.db")
cursor = conn.cursor()
sql = '''
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userId VARCHAR(255) NOT NULL,
    displayName VARCHAR(255) NOT NULL,
    request TEXT NOT NULL,
    emotion VARCHAR(10) NOT NULL,
    response TEXT NULL,
    created_at DATETIME NOT NULL
)
'''
cursor.execute(sql)
conn.commit()
conn.close()