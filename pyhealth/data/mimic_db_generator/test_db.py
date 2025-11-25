"""Script to test the MIMIC-III database files.

This script tests both the sampled database (mimic.db) and the full database
(mimic_all.db) by displaying their table structures and sample data.
"""

import sqlite3
import pandas as pd


def test_database(db_name):
    """Test a specific database file.

    Args:
        db_name (str): Name of the database file to test.

    This function:
    1. Connects to the database
    2. Lists all available tables
    3. Displays the first 5 rows of each table
    4. Shows the column names for each table
    """
    print(f"\n{'='*50}")
    print(f"Testing database: {db_name}")
    print(f"{'='*50}")
    
    # Connect to the database
    conn = sqlite3.connect(f'output/{db_name}')
    
    # Get all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables in the database:")
    for table in tables:
        print(f"- {table[0]}")
    
    # Test basic queries for each table
    for table in tables:
        table_name = table[0]
        print(f"\nTesting table {table_name}:")
        try:
            # Get first 5 rows
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
            print(f"First 5 rows of table {table_name}:")
            print(df)
            print(f"\nColumns in table {table_name}:")
            print(df.columns.tolist())
        except Exception as e:
            print(f"Error querying table {table_name}: {str(e)}")
    
    # Close connection
    conn.close()


if __name__ == "__main__":
    # Test both databases
    test_database('mimic.db')
    test_database('mimic_all.db') 