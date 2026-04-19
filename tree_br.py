import numpy as np
import psycopg2
import json
import pickle


# ==========================================
# 1. ĐỊNH NGHĨA CẤU TRÚC KD-TREE
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


def build_kdtree(points, ids, depth=0):
    if len(points) == 0:
        return None
    k = len(points[0])
    axis = depth % k
    combined = list(zip(points, ids))
    combined.sort(key=lambda x: x[0][axis])
    median_idx = len(combined) // 2
    median_point, median_id = combined[median_idx]

    left_branch = build_kdtree(
        [x[0] for x in combined[:median_idx]],
        [x[1] for x in combined[:median_idx]], depth + 1)
    right_branch = build_kdtree(
        [x[0] for x in combined[median_idx + 1:]],
        [x[1] for x in combined[median_idx + 1:]], depth + 1)

    return KDNode(median_point, median_id, left_branch, right_branch, axis)


# ==========================================
# 2. KHỞI TẠO HỆ THỐNG & KẾT NỐI DATABASE
# ==========================================
DB_PARAMS = {
    "host": "localhost", "port": "5432",
    "dbname": "speech",
    "user": "postgres", "password": "123456"
}

print("1. Đang kết nối CSDL PostgreSQL...")
try:
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    # Lấy 500 vector từ Database
    print("2. Đang kéo dữ liệu MFCC từ Database...")
    cursor.execute("SELECT audio_id, file_name, mfcc FROM Audio;")
    records = cursor.fetchall()

    points_from_db = []
    ids_from_db = []
    audio_dict = {}  # Dictionary để tra cứu tên file nhanh từ ID

    for rec in records:
        a_id, f_name, mfcc_str = rec
        audio_dict[a_id] = f_name

        # Chuyển đổi chuỗi '[0.1, 0.2]' từ DB thành numpy array và Chuẩn hóa L2
        mfcc_vector = np.array(json.loads(mfcc_str))
        points_from_db.append(normalize_vector(mfcc_vector))
        ids_from_db.append(a_id)

    # Đóng kết nối DB vì đã lấy đủ dữ liệu cần thiết
    cursor.close()
    conn.close()

    # ==========================================
    # 3. DỰNG CÂY VÀ LƯU FILE NHỊ PHÂN
    # ==========================================
    print("3. Đang dựng cấu trúc KD-Tree trên RAM...")
    kd_tree_root = build_kdtree(points_from_db, ids_from_db)

    # Gom cả Cây và Từ điển (map ID -> Tên file) vào chung một gói để lưu
    # Việc lưu kèm audio_dict giúp file truy vấn sau này không cần gọi lại DB nữa
    model_data = {
        "tree_root": kd_tree_root,
        "audio_dict": audio_dict
    }

    # Lưu xuống ổ cứng thành file nhị phân (.pkl)
    output_file = "kdtree_model.pkl"
    print(f"4. Đang lưu mô hình xuống file '{output_file}'...")

    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)

    print("\n HOÀN THÀNH! File mô hình đã sẵn sàng cho hệ thống truy vấn.")

except Exception as e:
    print(f"\n Lỗi xảy ra: {e}")