import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "aquaguard.db")


def create_database():
    connection=sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ph REAL,
            hardness REAL,
            solids REAL,
            chloramines REAL,
            sulfate REAL,
            conductivity REAL,
            organic_carbon REAL,
            trihalomethanes REAL,
            turbidity REAL,
            potability_risk TEXT,
            potability_probability REAL,
            anomaly_status TEXT,
            overall_risk TEXT,
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()

def save_analysis(
    ph,
    hardness,
    solids,
    chloramines,
    sulfate,
    conductivity,
    organic_carbon,
    trihalomethanes,
    turbidity,
    potability_risk,
    potability_probability,
    anomaly_status,
    overall_risk,
    recommendation
):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO analyses (
            ph,
            hardness,
            solids,
            chloramines,
            sulfate,
            conductivity,
            organic_carbon,
            trihalomethanes,
            turbidity,
            potability_risk,
            potability_probability,
            anomaly_status,
            overall_risk,
            recommendation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity,
        potability_risk,
        potability_probability,
        anomaly_status,
        overall_risk,
        recommendation
    ))

    connection.commit()
    connection.close()