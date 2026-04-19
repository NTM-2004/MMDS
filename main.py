import os
import pandas as pd
import numpy as np
import librosa
import whisper  # <-- Thêm thư viện Whisper
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Cấu hình đường dẫn ---
input_folder = r"D:\College\mmds\test"
output_excel = "audio_features_report.xlsx"

print("--- Đang tải mô hình Speech-to-Text Offline ---")
# Khởi tạo Whisper (Chạy 1 lần duy nhất)
# Các kích cỡ: 'tiny' (rất nhẹ), 'base' (cân bằng), 'small' (chậm hơn nhưng rất chuẩn)
stt_model = whisper.load_model("base")

data_list = []
print("--- Bắt đầu xử lý dữ liệu ---")


# --- 2. Xử lý từng file để lấy MFCC và Transcripts ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".mp3") or filename.lower().endswith(".wav"):
        file_path = os.path.join(input_folder, filename)
        print(f"Đang trích xuất: {filename}")

        try:
            # A. Trích xuất đặc trưng MFCC (Âm thanh)
            y, sr_rate = librosa.load(file_path)
            mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs.T, axis=0)

            # B. Chuyển đổi Speech-to-Text (Nội dung) bằng Whisper Offline
            # Trực tiếp truyền đường dẫn file vào mô hình.
            # Có thể thêm tham số language="en" hoặc "vi" để thuật toán không tốn thời gian tự đoán ngôn ngữ.
            transcribe_result = stt_model.transcribe(file_path, language="en", fp16=False)
            text_content = transcribe_result["text"].strip()

            # Lưu dữ liệu thô vào danh sách
            data_list.append({
                "file_name": filename,
                "mfcc_vector": mfcc_mean,
                "transcript": text_content if text_content else "N/A"
            })

        except Exception as e:
            print(f"Lỗi tại file {filename}: {e}")

# --- 3. Tổng hợp và xuất ra Excel ---
# LƯU Ý: Không cần lưu vector TF-IDF vào Excel để tránh lỗi quá tải số lượng ký tự trong một ô.
# Việc tính toán TF-IDF sẽ được thực hiện trực tiếp ở bước đưa dữ liệu vào PostgreSQL.

final_data = []
for item in data_list:
    row = {
        "File_Name": item["file_name"],
        "Transcript": item["transcript"],
        "MFCC_Mean_Vector": str(item["mfcc_vector"].tolist()) # Chỉ lưu MFCC (13 chiều) thì rất nhẹ
    }
    final_data.append(row)

df = pd.DataFrame(final_data)
df.to_excel(output_excel, index=False)

print(f"\n--- Hoàn thành! File dữ liệu thô đã được lưu tại: {output_excel} ---")