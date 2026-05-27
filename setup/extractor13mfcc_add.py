import os
import pandas as pd
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "merged_audio"   
OUTPUT_EXCEL = "audio_features_report.xlsx"
SAMPLE_RATE = 16000           
N_MFCC = 13                   
N_FFT = 512 
HOP_LENGTH = 160

def extract_features(file_path):
    try:
        y_raw, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 1. Pre-Silence Feature: Silence Ratio
        rms_raw = librosa.feature.rms(y=y_raw, frame_length=N_FFT, hop_length=HOP_LENGTH)
        silence_ratio = np.sum(rms_raw < 0.01) / float(rms_raw.shape[1])
        
        # 2. Silence Removal
        intervals = librosa.effects.split(y_raw, top_db=20)
        if len(intervals) == 0:
            return None
        y_clean = np.concatenate([y_raw[start:end] for start, end in intervals])

        # 3. Clean Audio Features
        mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        centroid = librosa.feature.spectral_centroid(y=y_clean, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        centroid_mean = np.mean(centroid)
        
        y_harmonic, _ = librosa.effects.hpss(y_clean)
        harmonicity = np.mean(librosa.feature.rms(y=y_harmonic, frame_length=N_FFT, hop_length=HOP_LENGTH))

        # 4. Combine into 16-D vector
        raw_vector = np.concatenate((mfcc_mean, [silence_ratio, harmonicity, centroid_mean]))
        return raw_vector
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    raw_data = []
    print("--- 1. Extracting Raw Features ---")
    
    for filename in os.listdir(AUDIO_DIR):
        if filename.lower().endswith((".mp3", ".wav", ".flac")):
            print(f"Processing: {filename}")
            filepath = os.path.join(AUDIO_DIR, filename)
            vec = extract_features(filepath)
            if vec is not None:
                raw_data.append({"File_Name": filename, "Raw_Vector": vec})

    if not raw_data:
        print("No files processed.")
        return

    print("--- 2. Applying Z-Score & L2 Normalization ---")
    all_vectors = np.array([item["Raw_Vector"] for item in raw_data])
    
    # Z-Score Standardization (Balances the features)
    means = np.mean(all_vectors, axis=0)
    stds = np.std(all_vectors, axis=0) + 1e-6 
    standardized = (all_vectors - means) / stds
    
    final_data = []
    for i, item in enumerate(raw_data):
        vec = standardized[i]
        
        # L2 Normalization (Required for KD-Tree)
        norm = np.linalg.norm(vec)
        l2_vector = vec if norm == 0 else vec / norm
        
        final_data.append({
            "File_Name": item["File_Name"],
            "Transcript": "N/A",
            "MFCC_Mean_Vector": str(l2_vector.tolist())
        })

    df = pd.DataFrame(final_data)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Done! Saved {len(final_data)} perfectly formatted 16-D vectors to Excel.")

if __name__ == "__main__":
    main()