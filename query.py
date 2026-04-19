import numpy as np
import psycopg2
import librosa
import whisper
import pickle
import os


# ==========================================
# 1. ĐỊNH NGHĨA LẠI CẤU TRÚC VÀ HÀM TÌM KIẾM
# (Bắt buộc phải có để đọc file .pkl)
# ==========================================
class KDNode:
    def __init__(self, point, audio_id, left=None, right=None, axis=0):
        self.point = point
        self.audio_id = audio_id
        self.left = left
        self.right = right
        self.axis = axis


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def search_kdtree(root, target, k=3):
    best_nodes = []  # Lưu danh sách (khoảng_cách, KDNode)

    def search(node):
        if node is None: return

        dist = euclidean_distance(target, node.point)
        best_nodes.append((dist, node))
        best_nodes.sort(key=lambda x: x[0])
        if len(best_nodes) > k:
            best_nodes.pop()

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
# 2. KHỞI TẠO VÀ TẢI MÔ HÌNH LÊN RAM
# ==========================================
print("1. Đang tải cấu trúc KD-Tree từ file nhị phân...")
model_file = "kdtree_model.pkl"

if not os.path.exists(model_file):
    print(f"❌ Không tìm thấy '{model_file}'. Vui lòng chạy file build_tree.py trước!")
    exit()

with open(model_file, "rb") as f:
    model_data = pickle.load(f)

kd_tree_root = model_data["tree_root"]
audio_dict = model_data["audio_dict"]

print("2. Đang tải mô hình nhận diện giọng nói (Whisper)...")
stt_model = whisper.load_model("base")

print("3. Đang kết nối CSDL PostgreSQL (cho truy vấn nội dung)...")
DB_PARAMS = {
    "host": "localhost", "port": "5432",
    "dbname": "speech",
    "user": "postgres", "password": "123456"
}
conn = psycopg2.connect(**DB_PARAMS)
cursor = conn.cursor()

# ==========================================
# 3. NHẬN FILE ĐẦU VÀO VÀ TRÍCH XUẤT ĐẶC TRƯNG
# ==========================================
print("\n" + "-" * 50)
input_file = r"D:\College\mmds\slice\CS50 Lecture by Mark Zuckerberg - 7 December 2005 [xFFs9UgOAlE]_part_20.mp3"

if not os.path.exists(input_file):
    print("❌ Lỗi: File âm thanh không tồn tại. Vui lòng kiểm tra lại đường dẫn!")
    exit()

print("Đang phân tích file đầu vào...")

# A. Trích xuất MFCC (Giọng nói)
y, sr_rate = librosa.load(input_file)
mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
input_mfcc = np.mean(mfccs.T, axis=0)
input_mfcc_norm = normalize_vector(input_mfcc)

# B. Trích xuất Văn bản (Nội dung)
transcribe_result = stt_model.transcribe(input_file, language="en")
input_text = transcribe_result["text"].lower()
input_words = tuple(set(input_text.split()))

print(f"-> Whisper nhận diện nội dung: '{input_text}'")

# ==========================================
# 4. TRUY VẤN VÀ HIỂN THỊ KẾT QUẢ
# ==========================================
# print("\n" + "=" * 50)
# print("🚀 KẾT QUẢ TÌM KIẾM THEO ĐẶC TRƯNG GIỌNG NÓI (VOICE)")
# print("   (Truy vấn qua KD-Tree trên RAM)")
# print("=" * 50)
#
# # Chạy thuật toán tìm 3 hàng xóm gần nhất trên cây
# top_3_voice = search_kdtree(kd_tree_root, input_mfcc_norm, k=3)
# for rank, (dist, node) in enumerate(top_3_voice, 1):
#     f_name = audio_dict[node.audio_id]
#     print(f"Top {rank}: {f_name} | Khoảng cách: {dist:.4f}")

print("\n" + "=" * 50)
print("🚀 KẾT QUẢ TÌM KIẾM THEO NỘI DUNG (CONTENT)")
print("   (Truy vấn qua PostgreSQL Inverted Index)")
print("=" * 50)

if input_words:
    # Gửi lệnh lên CSDL để tính điểm TF-IDF tổng cộng
    query = """
        SELECT a.file_name, SUM(i.tf_idf) as match_score
        FROM InvertedFile i
        JOIN Keyword k ON i.keyword_id = k.keyword_id
        JOIN Audio a ON i.audio_id = a.audio_id
        WHERE k.word IN %s
        GROUP BY a.file_name
        ORDER BY match_score DESC
        LIMIT 3;
    """
    cursor.execute(query, (input_words,))
    top_3_content = cursor.fetchall()

    if top_3_content:
        for rank, rec in enumerate(top_3_content, 1):
            f_name, score = rec
            print(f"Top {rank}: {f_name} | Điểm tương đồng: {score:.4f}")
    else:
        print("Không tìm thấy file nào trong hệ thống khớp với nội dung này.")
else:
    print("Không nhận diện được từ vựng nào trong file đầu vào.")

# Dọn dẹp kết nối
cursor.close()
conn.close()