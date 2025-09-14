import csv
from datetime import datetime, timedelta

def create_fake_los_data(filename="fake_los_data.csv"):
    # Define headers expected by the LOS task
    headers = ['admission_id', 'patient_id', 'admission_time', 'discharge_time']

    # Prepare some example rows
    base_date = datetime(2024, 1, 1, 10, 0)
    rows = [
        ['1', '1001', base_date.strftime("%Y-%m-%d %H:%M:%S"), (base_date + timedelta(days=4, hours=2)).strftime("%Y-%m-%d %H:%M:%S")],
        ['2', '1002', (base_date + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), (base_date + timedelta(days=3, hours=5)).strftime("%Y-%m-%d %H:%M:%S")],
        ['3', '1003', (base_date + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"), (base_date + timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")],
    ]

    # Write to CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Created fake LOS data CSV file: {filename}")

if __name__ == "__main__":
    create_fake_los_data()
