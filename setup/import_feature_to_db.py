import pandas as pd
import psycopg2
import ast
from pgvector.psycopg2 import register_vector

# --- CẤU HÌNH DATABASE ---
DB_HOST = "localhost"
DB_NAME = "speech"
DB_USER = "postgres"  # Thay bằng username của bạn
DB_PASS = "123456"  # Thay bằng mật khẩu của bạn

EXCEL_FILE = "audio_features_report.xlsx"


def main():
    print(f"--- 1. Đang đọc dữ liệu từ {EXCEL_FILE} ---")
    try:
        df = pd.read_excel(EXCEL_FILE)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {EXCEL_FILE}. Hãy chắc chắn bạn đã chạy script trích xuất trước.")
        return

    print("--- 2. Đang kết nối tới PostgreSQL ---")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )

        # Đăng ký kiểu dữ liệu vector của pgvector cho psycopg2 hiểu
        register_vector(conn)
        cur = conn.cursor()
        print("Kết nối thành công!\n")

        print("--- 3. Đang chèn dữ liệu vào bảng Audio ---")
        inserted_count = 0

        for index, row in df.iterrows():
            file_name = row['File_Name']

            # Cột MFCC_Mean_Vector trong Excel đang là một chuỗi (String) dạng "[0.1, 0.2, ...]"
            # ast.literal_eval giúp chuyển chuỗi đó thành danh sách (List) thực sự trong Python
            vector_list = ast.literal_eval(row['MFCC_Mean_Vector'])

            # Đường dẫn file giả định (bạn có thể tuỳ chỉnh lại)
            file_path = f"merged_audio/{file_name}"

            # Câu lệnh SQL Insert
            # Chú ý: Không cần truyền audioId vì nó là SERIAL (tự động tăng)
            insert_query = """
                INSERT INTO Audio (fileName, filePath, speakerFeature)
                VALUES (%s, %s, %s);
            """

            cur.execute(insert_query, (file_name, file_path, vector_list))
            inserted_count += 1
            print(f"Đã chèn: {file_name}")

        # Xác nhận thay đổi và đóng kết nối
        conn.commit()
        cur.close()
        conn.close()

        print(f"\n✅ HOÀN TẤT! Đã đẩy thành công {inserted_count} bản ghi vào Database.")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình thao tác với DB: {e}")


if __name__ == "__main__":
    main()