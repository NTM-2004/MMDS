import os
from pydub import AudioSegment

# Đường dẫn gốc đến thư mục chứa các file cần gộp
root_dir = r"D:\College\mmds\test"
output_base_dir = r"D:\College\mmds"

# Đảm bảo thư mục đầu ra tồn tại
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)


def merge_all_audio_in_folder(input_folder, output_folder, output_filename="merged_all_test.flac"):
    print(f"--- BẮT ĐẦU QUÉT TẤT CẢ FILE TRONG: {input_folder} ---")

    # Khởi tạo một đoạn âm thanh trống
    combined_audio = AudioSegment.empty()
    file_count = 0

    # os.walk sẽ tự động duyệt qua toàn bộ thư mục gốc và các thư mục con bên trong nó
    for dirpath, dirnames, filenames in os.walk(input_folder):
        # Lọc các file âm thanh (mình để sẵn flac, wav, mp3 cho linh hoạt) và sắp xếp theo tên
        audio_files = sorted([f for f in filenames if f.endswith(('.flac', '.wav', '.mp3'))])

        for audio_file in audio_files:
            file_path = os.path.join(dirpath, audio_file)
            print(f"  -> Đang nối: {file_path}")

            try:
                # Đọc và ghép nối file
                audio_segment = AudioSegment.from_file(file_path)
                combined_audio += audio_segment
                file_count += 1
            except Exception as e:
                print(f"❌ Lỗi khi đọc file {file_path}: {e}")

    if file_count == 0:
        print("⚠️ Không tìm thấy file âm thanh nào trong thư mục này.")
        return

    # Xuất file tổng
    output_path = os.path.join(output_folder, output_filename)
    print(f"\n--- ĐANG XUẤT FILE TỔNG (Gộp từ {file_count} file) ---")
    print("Quá trình này có thể mất vài phút nếu dữ liệu lớn, vui lòng đợi...")

    # Xuất ra định dạng flac
    combined_audio.export(output_path, format="flac")
    print(f"🎉 HOÀN TẤT! File tổng đã được lưu tại: {output_path}")


if __name__ == "__main__":
    merge_all_audio_in_folder(root_dir, output_base_dir)