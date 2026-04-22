import sqlite3
import json
class StorageDB:
    def __init__(self, db_path="seo.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT,
            phrase TEXT,
            ws REAL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS competitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT,
            url TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT,
            block TEXT,
            content TEXT
        )
        """)

        self.conn.commit()

    # 🔑 ключи
    def add_keywords(self, project_id, keywords, ws_values):
        cursor = self.conn.cursor()
        for k, w in zip(keywords, ws_values):
            cursor.execute(
                "INSERT INTO keywords (project_id, phrase, ws) VALUES (?, ?, ?)",
                (project_id, k, float(w) if w else 0)
            )
        self.conn.commit()

    # 🌐 конкуренты
    def add_competitors(self, project_id, urls):
        cursor = self.conn.cursor()
        for url in urls:
            cursor.execute(
                "INSERT INTO competitors (project_id, url) VALUES (?, ?)",
                (project_id, url)
            )
        self.conn.commit()

    # ✍️ тексты блоков
    def add_text(self, project_id, block, content):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO texts (project_id, block, content) VALUES (?, ?, ?)",
            (project_id, block, content)
        )
        self.conn.commit()

    # 📦 получить всё по проекту (если нужно)
    def get_project_data(self, project_id):
        cursor = self.conn.cursor()

        cursor.execute("SELECT phrase, ws FROM keywords WHERE project_id=?", (project_id,))
        keywords = cursor.fetchall()

        cursor.execute("SELECT url FROM competitors WHERE project_id=?", (project_id,))
        competitors = [r[0] for r in cursor.fetchall()]

        cursor.execute("SELECT block, content FROM texts WHERE project_id=?", (project_id,))
        texts = cursor.fetchall()

        return {
            "keywords": keywords,
            "competitors": competitors,
            "texts": texts
        }