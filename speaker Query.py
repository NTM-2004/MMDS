import os
import numpy as np
import librosa
import pickle
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 0. ĐỊNH NGHĨA LỚP KDNode (Bắt buộc để load pkl)
# ==========================================
class KDNode:
    def __init__(self, point, audio_id, left=None, right=None, axis=0):
        self.point = point
        self.audio_id = audio_id
        self.left = left
        self.right = right
        self.axis = axis


# ==========================================
# 1. CẤU HÌNH & TÊN ĐẶC TRƯNG
# ==========================================
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160

FEATURE_NAMES = [
    "MFCC 1 (Overall Energy Envelope)",
    "MFCC 2 (Broad Vocal Tract Shape)",
    "MFCC 3 (Tongue Position/Vowels)",
    "MFCC 4", "MFCC 5", "MFCC 6", "MFCC 7", "MFCC 8",
    "MFCC 9", "MFCC 10", "MFCC 11", "MFCC 12", "MFCC 13",
    "Silence Ratio (Speech Pacing)",
    "Harmonicity (Resonance vs. Raspiness)",
    "Spectral Centroid (Voice Brightness/Tone)"
]


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def search_kdtree(root, target, k=3):
    best_nodes = []

    def search(node):
        if node is None: return
        dist = euclidean_distance(target, node.point)
        best_nodes.append((dist, node))
        best_nodes.sort(key=lambda x: x[0])
        if len(best_nodes) > k: best_nodes.pop()

        axis = node.axis
        diff = target[axis] - node.point[axis]
        close_branch = node.left if diff < 0 else node.right
        far_branch = node.right if diff < 0 else node.left

        search(close_branch)
        if len(best_nodes) < k or abs(diff) < best_nodes[-1][0]:
            search(far_branch)

    search(root)
    return best_nodes


# ==========================================
# 2. HÀM TRÍCH XUẤT VÀ CHUẨN HÓA (Dùng Scaler Pkl)
# ==========================================
def extract_and_normalize(file_path, means, stds):
    try:
        # A. Trích xuất thô (giống hệt extractor ban đầu)
        y_raw, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        rms_raw = librosa.feature.rms(y=y_raw, frame_length=N_FFT, hop_length=HOP_LENGTH)
        silence_ratio = np.sum(rms_raw < 0.01) / float(rms_raw.shape[1])

        intervals = librosa.effects.split(y_raw, top_db=20)
        if len(intervals) == 0: return None
        y_clean = np.concatenate([y_raw[start:end] for start, end in intervals])

        mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfccs, axis=1)

        centroid = librosa.feature.spectral_centroid(y=y_clean, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        centroid_mean = np.mean(centroid)

        y_harmonic, _ = librosa.effects.hpss(y_clean)
        harmonicity = np.mean(librosa.feature.rms(y=y_harmonic, frame_length=N_FFT, hop_length=HOP_LENGTH))

        raw_vector = np.concatenate((mfcc_mean, [silence_ratio, harmonicity, centroid_mean]))

        # B. Áp dụng Z-Score dựa trên tham số đã lưu
        standardized = (raw_vector - means) / stds

        # C. Áp dụng L2 Normalization
        norm = np.linalg.norm(standardized)
        l2_vector = standardized if norm == 0 else standardized / norm

        return l2_vector

    except Exception as e:
        print(f"Error: {e}")
        return None


# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
   model_file = "local_kdtree_model.pkl"
   scaler_file = "scaler_params.pkl"


   if not os.path.exists(model_file) or not os.path.exists(scaler_file):
       print("Error: Missing .pkl file!")
       return


   # Nạp dữ liệu
   print("[1/3] Push Tree & parameters into RAM...")
   with open(model_file, "rb") as f:
       model_data = pickle.load(f)
   with open(scaler_file, "rb") as f:
       scaler_data = pickle.load(f)


   kd_tree_root = model_data["tree_root"]
   audio_dict = model_data["audio_dict"]
   GLOBAL_MEANS = scaler_data["means"]
   GLOBAL_STDS = scaler_data["stds"]


   # Nhập file truy vấn
   query_audio_path = r"D:\College\mmds\6147_34605_first10.flac"
   if query_audio_path.startswith(('"', "'")) and query_audio_path.endswith(('"', "'")):
       query_audio_path = query_audio_path[1:-1]


   if not os.path.exists(query_audio_path):
       print("Error: None exist file")
       return


   # Xử lý
   print(f"[3/3] Analize voice...")
   query_vector = extract_and_normalize(query_audio_path, GLOBAL_MEANS, GLOBAL_STDS)


   if query_vector is None: return


   # In kết quả theo phong cách test_query.py
   print(f"\n{'=' * 80}")
   print(f"SEARCHING KD-TREE FOR: {os.path.basename(query_audio_path)}")
   print(f"{'=' * 80}\n")


   top_results = search_kdtree(kd_tree_root, query_vector, k=3)


   for rank, (dist, node) in enumerate(top_results, 1):
       f_name = audio_dict.get(node.audio_id, "Unknown File")


       # Công thức tính % tương đồng từ khoảng cách Euclide
       similarity_pct = max(0.0, (1 - (dist ** 2) / 2)) * 100


       # Phân tích các đặc trưng khớp nhất
       feature_diffs = np.abs(query_vector - node.point)
       closest_indices = np.argsort(feature_diffs)[:3]
       driving_features = [FEATURE_NAMES[i] for i in closest_indices]


       print(f"[{rank}] {f_name}")
       print(f"    ├─ Similarity: {similarity_pct:.2f}%  (Distance: {dist:.4f})")
       print(f"    └─ Strongest Matches: {driving_features[0]}, {driving_features[1]}, {driving_features[2]}\n")



if __name__ == "__main__":
    main()