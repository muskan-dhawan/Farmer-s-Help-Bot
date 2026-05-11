import sqlite3
import time

class DatabaseManager:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.create_table()

    # 🔥 FINAL TABLE STRUCTURE (WITH name FIX)
    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            state TEXT,
            lat REAL,
            lon REAL,
            lang TEXT,
            mode TEXT,
            last_crop TEXT,
            timestamp REAL
        )
        """)

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON users(timestamp)"
        )

    # 🔥 UPSERT WITH TIMESTAMP (NO BINDING ERROR)
    def upsert_user(self, user_id, **kwargs):
        kwargs["timestamp"] = time.time()  # always update timestamp

        fields = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        updates = ", ".join([f"{k}=excluded.{k}" for k in kwargs.keys()])

        query = f"""
        INSERT INTO users (user_id, {fields})
        VALUES (?, {placeholders})
        ON CONFLICT(user_id) DO UPDATE SET {updates}
        """

        values = [user_id] + list(kwargs.values())

        self.conn.execute(query, values)
        self.conn.commit()

    # 🔥 GET USER WITH AUTO EXPIRY (24H DEFAULT)
    def get_user(self, user_id, expiry_hours=24):
        cur = self.conn.execute(
            "SELECT * FROM users WHERE user_id=?",
            (user_id,)
        )
        row = cur.fetchone()

        if not row:
            return None

        cols = [c[0] for c in cur.description]
        user = dict(zip(cols, row))

        # 🔥 EXPIRY CHECK
        now = time.time()
        last = user.get("timestamp") or 0

        if last and (now - last) > expiry_hours * 3600:
            self.conn.execute(
                "DELETE FROM users WHERE user_id=?",
                (user_id,)
            )
            self.conn.commit()
            return None

        return user

    # 🔥 CLEANUP OLD USERS (48H)
    def cleanup_expired(self, expiry_hours=48):
        cutoff = time.time() - expiry_hours * 3600
        self.conn.execute(
            "DELETE FROM users WHERE timestamp < ?",
            (cutoff,)
        )
        self.conn.commit()