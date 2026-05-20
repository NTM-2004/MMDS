# import pickle
# import numpy as np
# import os
# import librosa
# import warnings
#
# # Tắt các cảnh báo không cần thiết từ librosa
# warnings.filterwarnings("ignore", category=UserWarning)
#
#
# # ==========================================
# # 0. ĐỊNH NGHĨA LỚP KDNode (BẮT BUỘC PHẢI CÓ)
# # ==========================================
# class KDNode:
#     def __init__(self, point, audio_id, left=None, right=None, axis=0):
#         self.point = point
#         self.audio_id = audio_id
#         self.left = left
#         self.right = right
#         self.axis = axis
#
#
# # --- CẤU HÌNH ---
# SAMPLE_RATE = 16000
# N_MFCC = 13
# N_FFT = 512
# HOP_LENGTH = 160
# AUDIO_DIR = r"D:\College\mmds\merged_audio"
# MODEL_FILE = "local_kdtree_model.pkl"
# SCALER_FILE = "scaler_params.pkl"
#
#
# def extract_raw_features(file_path):
#     try:
#         y_raw, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#
#         # 1. Silence Ratio
#         rms_raw = librosa.feature.rms(y=y_raw, frame_length=N_FFT, hop_length=HOP_LENGTH)
#         silence_ratio = np.sum(rms_raw < 0.01) / float(rms_raw.shape[1])
#
#         # 2. Silence Removal
#         intervals = librosa.effects.split(y_raw, top_db=20)
#         if len(intervals) == 0:
#             return None
#         y_clean = np.concatenate([y_raw[start:end] for start, end in intervals])
#
#         # 3. Clean Audio Features
#         mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
#         mfcc_mean = np.mean(mfccs, axis=1)
#
#         centroid = librosa.feature.spectral_centroid(y=y_clean, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
#         centroid_mean = np.mean(centroid)
#
#         y_harmonic, _ = librosa.effects.hpss(y_clean)
#         harmonicity = np.mean(librosa.feature.rms(y=y_harmonic, frame_length=N_FFT, hop_length=HOP_LENGTH))
#
#         return np.concatenate((mfcc_mean, [silence_ratio, harmonicity, centroid_mean]))
#
#     except Exception as e:
#         print(f"Lỗi khi xử lý {file_path}: {e}")
#         return None
#
#
# def main():
#     print(f"--- 1. Đang đọc danh sách file từ {MODEL_FILE} ---")
#     if not os.path.exists(MODEL_FILE):
#         print("Lỗi: Không tìm thấy file model pkl!")
#         return
#
#     # Pickle sẽ không còn báo lỗi vì đã có định nghĩa KDNode ở trên
#     with open(MODEL_FILE, "rb") as f:
#         model_data = pickle.load(f)
#
#     audio_dict = model_data["audio_dict"]
#     filenames = list(audio_dict.values())
#
#     print(f"--- 2. Đang trích xuất đặc trưng thô từ {len(filenames)} file ---")
#     raw_vectors = []
#     for fname in filenames:
#         fpath = os.path.join(AUDIO_DIR, fname)
#         vec = extract_raw_features(fpath)
#         if vec is not None:
#             raw_vectors.append(vec)
#
#     if not raw_vectors:
#         print("Không có dữ liệu để tính toán!")
#         return
#
#     all_vectors = np.array(raw_vectors)
#     means = np.mean(all_vectors, axis=0)
#     stds = np.std(all_vectors, axis=0) + 1e-6
#
#     print(f"--- 3. Đang lưu tham số vào {SCALER_FILE} ---")
#     scaler_data = {
#         "means": means,
#         "stds": stds
#     }
#
#     with open(SCALER_FILE, "wb") as f:
#         pickle.dump(scaler_data, f)
#
#     print("\n✅ HOÀN TẤT! File scaler_params.pkl đã được tạo thành công.")
#
#
# if __name__ == "__main__":
#     main()

import pickle
import numpy as np
import os
import librosa
import warnings

# Disable unnecessary warnings from librosa
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 0. KDNode CLASS DEFINITION (REQUIRED)
# ==========================================
class KDNode:
    def __init__(self, point, audio_id, left=None, right=None, axis=0):
        self.point = point
        self.audio_id = audio_id
        self.left = left
        self.right = right
        self.axis = axis


# --- CONFIGURATION ---
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
AUDIO_DIR = r"D:\College\mmds\merged_audio"
MODEL_FILE = "../local_kdtree_model.pkl"
SCALER_FILE = "../scaler_params.pkl"


def extract_raw_features(file_path):
    try:
        y_raw, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # 1. Silence Ratio
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

        return np.concatenate((mfcc_mean, [silence_ratio, harmonicity, centroid_mean]))

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    print(f"--- 1. Reading file list from {MODEL_FILE} ---")
    if not os.path.exists(MODEL_FILE):
        print("Error: Model pkl file not found!")
        return

    # Pickle will not throw an error because the KDNode class is defined above
    with open(MODEL_FILE, "rb") as f:
        model_data = pickle.load(f)

    audio_dict = model_data["audio_dict"]
    filenames = list(audio_dict.values())

    print(f"--- 2. Extracting raw features from {len(filenames)} files ---")
    raw_vectors = []
    for fname in filenames:
        fpath = os.path.join(AUDIO_DIR, fname)
        vec = extract_raw_features(fpath)
        if vec is not None:
            raw_vectors.append(vec)

    if not raw_vectors:
        print("No data available for calculation!")
        return

    all_vectors = np.array(raw_vectors)
    means = np.mean(all_vectors, axis=0)
    stds = np.std(all_vectors, axis=0) + 1e-6

    print(f"--- 3. Saving parameters to {SCALER_FILE} ---")
    scaler_data = {
        "means": means,
        "stds": stds
    }

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler_data, f)

    print("\nCOMPLETE! The scaler_params.pkl file has been created successfully.")


if __name__ == "__main__":
    main()