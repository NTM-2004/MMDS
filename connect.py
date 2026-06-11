import psycopg2
from psycopg2 import OperationalError

# Cấu hình kết nối dựa trên thông tin DB_PARAMS của bạn
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "speech",
    "user": "postgres",
    "password": "123456"
}


def connect_and_setup():
    connection = None
    try:
        # 1. Kết nối tới cơ sở dữ liệu PostgreSQL
        print("Đang kết nối tới PostgreSQL...")
        connection = psycopg2.connect(**DB_PARAMS)

        # Tạo cursor để thực thi các câu lệnh SQL
        cursor = connection.cursor()

        # Kiểm tra kết nối bằng cách lấy phiên bản của Postgres
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"🎉 Kết nối thành công! Phiên bản Postgres: {db_version[0]}")

        # 🔥 BƯỚC QUAN TRỌNG: Kích hoạt extension pgvector cho database này
        print("⚙️ Đang kích hoạt extension 'pgvector'...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Tạo các bảng cho hệ thống Audio Retrieval của bạn
        print("📊 Đang khởi tạo cấu trúc các bảng...")
        create_table_query = """
        delete from audio
        """
        cursor.execute(create_table_query)

        # Xác nhận (Commit) các thay đổi vào cơ sở dữ liệu
        connection.commit()
        print("✅ Đã khởi tạo cấu trúc database (Audio, Keyword, InvertedFile) thành công!")

        # Đóng cursor sau khi hoàn thành công việc
        cursor.close()

    except Exception as e:
        print(f"❌ Lỗi thực thi SQL: {e}")
        if connection:
            connection.rollback() # Quay xe nếu có lỗi xảy ra để tránh treo transaction
    finally:
        # Đảm bảo luôn đóng kết nối để giải phóng tài nguyên
        if connection is not None:
            connection.close()
            print("🔌 Đã đóng kết nối cơ sở dữ liệu an toàn.")


if __name__ == "__main__":
    connect_and_setup()