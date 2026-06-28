import sqlite3
import os

def create_db_indexes():
    db_path = "users.db"
    if not os.path.exists(db_path):
        print("Database users.db does not exist yet. Indexes will be created on start.")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        print("Setting up optimized database indexes on predictions_history table...")
        
        # Create compound index for fast location queries (district + station)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_location ON predictions_history(district, station)")
        
        # Create timestamp index for fast chronological sorting and auditing
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions_history(timestamp DESC)")
        
        conn.commit()
        conn.close()
        print("[OK] Optimized SQLite indexes created successfully!")
    except Exception as e:
        print("Error creating database indexes:", str(e))

if __name__ == "__main__":
    create_db_indexes()
