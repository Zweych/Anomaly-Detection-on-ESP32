import serial
import csv
import time
from datetime import datetime
import os

# Konfigurasi Port
SERIAL_PORT = 'COM7' 
BAUD_RATE = 115200
OUTPUT_FILE = 'bme280_june_data.csv'

# Membuka koneksi serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
except Exception as e:
    print(f"Gagal membuka {SERIAL_PORT}. Pastikan Serial Monitor di Arduino IDE ditutup.")
    exit()

# Cek apakah file sudah ada untuk menentukan penulisan header
file_exists = os.path.isfile(OUTPUT_FILE)

with open(OUTPUT_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Tulis header jika ini adalah perekaman pertama
    if not file_exists:
        writer.writerow(["datetime", "temperature", "pressure", "humidity"])

    print(f"Merekam data dari {SERIAL_PORT} ke {OUTPUT_FILE}... (Tekan Ctrl+C untuk berhenti)")
    
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Mengabaikan baris kosong atau pesan error
                if line and "Error" not in line:
                    data = line.split(',')
                    
                    if len(data) == 3:
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        row = [now] + data
                        
                        writer.writerow(row)
                        file.flush() # Memaksa sistem menulis data dari RAM ke penyimpanan disk
                        
                        print(f"Terekam: {row}")
    except KeyboardInterrupt:
        print("\nProses perekaman dihentikan dengan aman.")
        ser.close()