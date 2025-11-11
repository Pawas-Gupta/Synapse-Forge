# DATABASE_PY = """
# \"\"\"Database initialization and management\"\"\"
import sqlite3
import json
from typing import Optional

class Database:
    def __init__(self, db_path: str = ':memory:'):
        self.db_path = db_path
        self.conn = None
        self.initialize()
    
    def initialize(self):
        # \"\"\"Initialize database with schema and mock data\"\"\"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        self._insert_mock_data()
    
    def _create_tables(self):
        # \"\"\"Create database tables\"\"\"
        cursor = self.conn.cursor()
        
        # SKU Catalog Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS SKU_CATALOG (
                sku_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                specs TEXT,
                unit_cost REAL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # RFP Data Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS RFP_DATA (
                rfp_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT,
                requirements TEXT,
                status TEXT DEFAULT 'New',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # RFP Matches Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS RFP_MATCHES (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                rfp_id TEXT,
                sku_id TEXT,
                match_score REAL,
                quantity INTEGER,
                FOREIGN KEY (rfp_id) REFERENCES RFP_DATA(rfp_id),
                FOREIGN KEY (sku_id) REFERENCES SKU_CATALOG(sku_id)
            )
        ''')
        
        self.conn.commit()
    
    def _insert_mock_data(self):
        # \"\"\"Insert mock SKU data\"\"\"
        cursor = self.conn.cursor()
        
        mock_skus = [
            ('SKU-001', 'Industrial Circuit Breaker 100A', 
             'Heavy-duty circuit breaker for industrial applications',
             json.dumps({
                 'voltage': '480V',
                 'current': '100A',
                 'poles': 3,
                 'type': 'thermal-magnetic',
                 'interrupt_rating': '65kA',
                 'ul_listed': True
             }),
             245.50, 'Circuit Protection'),
            
            ('SKU-002', 'Industrial Circuit Breaker 200A',
             'High-capacity circuit breaker for heavy industrial use',
             json.dumps({
                 'voltage': '480V',
                 'current': '200A',
                 'poles': 3,
                 'type': 'thermal-magnetic',
                 'interrupt_rating': '65kA',
                 'ul_listed': True
             }),
             425.00, 'Circuit Protection'),
            
            ('SKU-003', 'Motor Starter 50HP',
             'Three-phase motor starter with overload protection',
             json.dumps({
                 'horsepower': '50HP',
                 'voltage': '480V',
                 'control': 'electronic',
                 'protection': 'overload',
                 'nema_size': 3
             }),
             890.00, 'Motor Control'),
            
            ('SKU-004', 'Distribution Panel 400A',
             'Main distribution panel for industrial facilities',
             json.dumps({
                 'current': '400A',
                 'voltage': '480V',
                 'buses': 'copper',
                 'spaces': 42,
                 'main_breaker': True
             }),
             1250.00, 'Distribution'),
            
            ('SKU-005', 'Transformer 75kVA',
             'Dry-type transformer for voltage conversion',
             json.dumps({
                 'capacity': '75kVA',
                 'primary': '480V',
                 'secondary': '208V',
                 'type': 'dry',
                 'efficiency': '98%'
             }),
             3200.00, 'Power Conversion'),
            
            ('SKU-006', 'Cable Tray System 12in',
             'Galvanized cable tray system for cable management',
             json.dumps({
                 'width': '12in',
                 'material': 'galvanized-steel',
                 'load': 'heavy-duty',
                 'length': '10ft',
                 'finish': 'pre-galvanized'
             }),
             185.00, 'Cable Management'),
            
            ('SKU-007', 'Emergency Lighting System',
             'Battery backup emergency lighting for safety compliance',
             json.dumps({
                 'battery': 'lithium',
                 'runtime': '90min',
                 'lumens': 1000,
                 'compliance': 'UL924',
                 'test_button': True
             }),
             320.00, 'Safety Equipment'),
            
            ('SKU-008', 'Conduit Set 2in EMT',
             'Electrical metallic tubing for wire protection',
             json.dumps({
                 'size': '2in',
                 'type': 'EMT',
                 'material': 'steel',
                 'length': '10ft',
                 'ul_listed': True
             }),
             45.00, 'Conduit & Raceway')
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO SKU_CATALOG 
            (sku_id, name, description, specs, unit_cost, category) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', mock_skus)
        
        self.conn.commit()
    
    def get_connection(self):
        # \"\"\"Get database connection\"\"\"
        return self.conn
    
    def close(self):
        # \"\"\"Close database connection\"\"\"
        if self.conn:
            self.conn.close()

# Global database instance
db_instance = None

def get_db():
    # \"\"\"Get or create database instance\"\"\"
    global db_instance
    if db_instance is None:
        db_instance = Database()
    return db_instance
# """