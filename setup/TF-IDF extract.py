import os
import psycopg2
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "speech",
    "user": "postgres",
    "password": "123456"
}

# Thay đổi đường dẫn này trỏ tới thư mục chứa các file audio của bạn
AUDIO_FOLDER = r"D:\College\mmds\merged_audio"


def main():
    print("1. Đang kết nối Cơ sở dữ liệu PostgreSQL...")
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    # Kéo danh sách file đã có trong bảng Audio để map fileName với audioId
    cursor.execute("SELECT audioId, fileName FROM Audio;")
    audio_records = cursor.fetchall()

    # Tạo dictionary map: fileName -> audioId
    db_audio_map = {rec[1]: rec[0] for rec in audio_records}

    if not db_audio_map:
        print("Bảng Audio trống! Vui lòng điền dữ liệu bảng Audio trước khi chạy script này.")
        return

    print("2. Đang tải mô hình ngôn ngữ Whisper (Speech-to-Text)...")
    model = whisper.load_model("base")  # Dùng base hoặc small tùy cấu hình máy

    transcripts = []
    audio_ids_for_tfidf = []

    # ==========================================
    # 2. QUÉT FILE & TRÍCH XUẤT VĂN BẢN
    # ==========================================
    print("\n3. Đang trích xuất văn bản từ Audio...")
    for file_name in os.listdir(AUDIO_FOLDER):
        if file_name.endswith((".flac", ".wav", ".mp3")):
            # Chỉ xử lý nếu file audio này đã tồn tại trong bảng Audio
            if file_name in db_audio_map:
                file_path = os.path.join(AUDIO_FOLDER, file_name)
                print(f"  -> Đang nghe và dịch: {file_name}")

                try:
                    # Chuyển đổi giọng nói thàn h văn bản
                    result = model.transcribe(file_path, language="en")
                    text = result["text"].strip()

                    if text:
                        transcripts.append(text)
                        audio_ids_for_tfidf.append(db_audio_map[file_name])
                except Exception as e:
                    print(f"     [Lỗi] Không thể xử lý {file_name}: {e}")

    if not transcripts:
        print("Không trích xuất được văn bản nào. Kết thúc chương trình.")
        return

    # ==========================================
    # 3. TÍNH TOÁN TF-IDF & LƯU DATABASE
    # ==========================================
    print("\n4. Đang tính toán ma trận TF-IDF cho toàn bộ kho dữ liệu...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(transcripts)
    feature_names = vectorizer.get_feature_names_out()

    print("\n5. Đang lưu Từ vựng & Chỉ mục ngược (Inverted Index) vào DB...")
    # Lặp qua từng đoạn văn bản của từng file
    for doc_idx, audio_id in enumerate(audio_ids_for_tfidf):
        doc_vector = tfidf_matrix[doc_idx]

        # Chỉ lặp qua các từ có xuất hiện trong file này (TF-IDF > 0)
        for col in doc_vector.nonzero()[1]:
            word = feature_names[col]
            score = float(doc_vector[0, col])

            # Bước A: Xử lý bảng Keyword
            # Mẹo SQL: Dùng ON CONFLICT... DO UPDATE để LUÔN LUÔN lấy được RETURNING keywordId
            # cho dù từ đó là mới chèn hay đã tồn tại từ trước.
            cursor.execute("""
                INSERT INTO Keyword (word) 
                VALUES (%s) 
                ON CONFLICT (word) DO UPDATE SET word = EXCLUDED.word
                RETURNING keywordId;
            """, (word,))
            keyword_id = cursor.fetchone()[0]

            # Bước B: Xử lý bảng InvertedFile
            cursor.execute("""
                INSERT INTO InvertedFile (audioId, keywordId, tf_idfScore) 
                VALUES (%s, %s, %s)
                ON CONFLICT (audioId, keywordId) DO UPDATE 
                SET tf_idfScore = EXCLUDED.tf_idfScore;
            """, (audio_id, keyword_id, score))

        print(f"  -> Đã lưu xong dữ liệu Index cho Audio ID: {audio_id}")

    # Commit thay đổi và đóng kết nối
    conn.commit()
    cursor.close()
    conn.close()
    print("\n🚀 HOÀN TẤT! Toàn bộ đặc trưng Content (TF-IDF) đã được tích hợp thành công vào PostgreSQL.")


if __name__ == "__main__":
    main()



