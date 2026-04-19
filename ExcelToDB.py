import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CẤU HÌNH DATABASE ---
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "speech"  # THAY ĐỔI TÊN DB CỦA BẠN
DB_USER = "postgres"  # THAY ĐỔI TÊN USER
DB_PASS = "123456"  # THAY ĐỔI MẬT KHẨU

# 1. Đọc dữ liệu thô từ file Excel
df = pd.read_excel("audio_features_report.xlsx")
df['Transcript'] = df['Transcript'].fillna("")

print("--- Bắt đầu tính toán phân tích TF-IDF ---")

# 2. Khởi tạo và huấn luyện TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Transcript'])
feature_names = vectorizer.get_feature_names_out()

# Mở kết nối đến PostgreSQL
print("--- Đang kết nối đến Database ---")
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()

    # 3. Quét từng file để bóc tách và đẩy vào DB
    for index, row in df.iterrows():
        file_name = row['File_Name']
        file_path = f"dataset/{file_name}"  # Lưu đường dẫn tương đối

        # Đảm bảo MFCC là chuỗi định dạng "[...]" để pgvector hiểu được
        mfcc_str = str(row['MFCC_Mean_Vector'])

        # --- A. LƯU BẢNG AUDIO ---
        # Bỏ qua id vì đã dùng SERIAL tự động tăng, ta dùng RETURNING để lấy id vừa tạo
        cursor.execute("""
            INSERT INTO Audio (file_name, file_path, mfcc) 
            VALUES (%s, %s, %s)
            RETURNING audio_id;
        """, (file_name, file_path, mfcc_str))

        audio_id = cursor.fetchone()[0]  # Lấy ID do DB tự cấp phát
        print(f"\n[Đã lưu File: {file_name} -> Audio ID: {audio_id}]")

        # Lấy vector TF-IDF của file hiện tại
        doc_vector = tfidf_matrix[index]

        # --- B. LƯU TỪ VỰNG VÀ CHỈ MỤC NGƯỢC ---
        for col in doc_vector.nonzero()[1]:
            word = feature_names[col]
            score = float(doc_vector[0, col])

            # B1. Xử lý bảng Keyword
            # Kiểm tra xem từ khóa đã tồn tại trong DB chưa
            cursor.execute("SELECT keyword_id FROM Keyword WHERE word = %s;", (word,))
            result = cursor.fetchone()

            if result:
                keyword_id = result[0]  # Nếu có rồi thì lấy ID
            else:
                # Nếu chưa có thì Insert từ mới và lấy ID trả về
                cursor.execute("""
                    INSERT INTO Keyword (word) 
                    VALUES (%s) 
                    RETURNING keyword_id;
                """, (word,))
                keyword_id = cursor.fetchone()[0]

            # B2. Xử lý bảng InvertedFile (Bảng trung gian)
            # Insert Audio ID, Keyword ID và điểm TF-IDF
            cursor.execute("""
                INSERT INTO InvertedFile (audio_id, keyword_id, tf_idf) 
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING; -- Tránh lỗi nếu bị chèn trùng lặp
            """, (audio_id, keyword_id, score))

    # Xác nhận toàn bộ thay đổi và lưu vào DB
    conn.commit()
    print("\n--- HOÀN THÀNH ĐƯA DỮ LIỆU VÀO DATABASE! ---")

except Exception as e:
    print(f"\n[Lỗi kết nối hoặc thực thi DB]: {e}")
    if 'conn' in locals():
        conn.rollback()  # Hoàn tác nếu có lỗi
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()