import pandas as pd
import numpy as np
import random

print("Memulai sintesis FDIA terkoreksi pada sensor BME280...")

df = pd.read_csv("/bme280_natural_outdoor.csv")
total_rows = len(df)
total_attack = total_rows // 2

num_classes = 8
max_ratio = 3

while True:
    weights = [random.uniform(1.0, 3.0) for _ in range(num_classes)]
    w_sum = sum(weights)
    dist = [int(w / w_sum * total_attack) for w in weights]
    dist[-1] += total_attack - sum(dist)
    if max(dist) / min(dist) <= max_ratio:
        break

target_counts = {i+1: dist[i] for i in range(num_classes)}

df['label_binary'] = 'normal'
df['label_multi'] = 'normal'

attack_names = {
    1: 'spike', 2: 'bias', 3: 'gaussian_noise', 4: 'replay',
    5: 'drift', 6: 'scaling', 7: 'freezing', 8: 'oscillation'
}

block_size = 100
total_blocks = total_rows // block_size

blocks_needed = {k: v // block_size for k, v in target_counts.items()}
attack_blocks = sum(blocks_needed.values())
normal_blocks = total_blocks - attack_blocks

assignment = [0] * normal_blocks
for k, v in blocks_needed.items():
    assignment.extend([k] * v)

random.shuffle(assignment)

# Menyimpan indeks blok normal untuk referensi Replay Attack yang aman
normal_block_indices = [i for i, x in enumerate(assignment) if x == 0]

for i, attack_class in enumerate(assignment):
    start_idx = i * block_size
    end_idx = start_idx + block_size
    
    if attack_class == 0:
        continue
        
    df.loc[start_idx:end_idx-1, 'label_binary'] = 'attack'
    df.loc[start_idx:end_idx-1, 'label_multi'] = attack_names[attack_class]
    
    if attack_class == 1:
        # Spike: Amplitudo dipertahankan, titik ditambah menjadi 10
        spike_indices = np.random.choice(range(start_idx, end_idx), size=10, replace=False)
        df.loc[spike_indices, 'temperature'] += np.random.choice([20.0, -20.0], size=10)
        
    elif attack_class == 2:
        # Bias: Ditambahkan micro-jitter agar diff tidak identik dengan normal
        jitter = np.random.normal(0, 0.5, block_size)
        df.loc[start_idx:end_idx-1, 'pressure'] += (25.0 + jitter)
        
    elif attack_class == 3:
        # Gaussian Noise: Standar deviasi dinaikkan agar lebih disruptif
        noise = np.random.normal(0, 6, block_size)
        df.loc[start_idx:end_idx-1, 'humidity'] += noise
        
    elif attack_class == 4:
        # Replay: Menyalin dari blok yang DIPASTIKAN normal
        if len(normal_block_indices) > 0:
            safe_normal_idx = random.choice(normal_block_indices)
            safe_start = safe_normal_idx * block_size
            safe_end = safe_start + block_size
            df.loc[start_idx:end_idx-1, 'temperature'] = df.loc[safe_start:safe_end-1, 'temperature'].values
            
    elif attack_class == 5:
        # Drift: Ditambahkan variansi acak di sepanjang garis linear
        drift = np.linspace(0, 15, block_size) + np.random.normal(0, 0.2, block_size)
        df.loc[start_idx:end_idx-1, 'temperature'] += drift
        
    elif attack_class == 6:
        # Scaling: Pengali diperbesar menjadi 1.5x
        df.loc[start_idx:end_idx-1, 'pressure'] *= 1.5
        
    elif attack_class == 7:
        # Freezing: (Tetap sama, karena diff = 0 sudah sangat terdeteksi)
        freeze_val = df.loc[start_idx, 'humidity']
        df.loc[start_idx:end_idx-1, 'humidity'] = freeze_val
        
    elif attack_class == 8:
        # Oscillation: Amplitudo dan frekuensi ditingkatkan
        t = np.arange(block_size)
        osc = 10 * np.sin(2 * np.pi * 0.25 * t)
        df.loc[start_idx:end_idx-1, 'temperature'] += osc

output_filename = "Train_Test_IoT_Weather_Natural_Fixed.csv"
df.to_csv(output_filename, index=False)
print(f"Dataset Master Terkoreksi berhasil disimpan sebagai {output_filename}")