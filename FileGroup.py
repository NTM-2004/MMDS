import os
from pydub import AudioSegment

# Đường dẫn gốc đến tập dữ liệu train-clean-100
# Thay đổi đường dẫn này cho đúng với máy của bạn
root_dir = "D:/College/mmds/train-clean-100/LibriSpeech/train-clean-100"
output_base_dir = "D:/College/mmds/merged_audio"

if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Duyệt qua từng thư mục người nói (speaker ID) [cite: 13, 17]
for speaker_id in os.listdir(root_dir):
    speaker_path = os.path.join(root_dir, speaker_id)

    if os.path.isdir(speaker_path):
        # Duyệt qua từng thư mục chương (chapter ID) của người nói đó [cite: 13, 18]
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)

            if os.path.isdir(chapter_path):
                # Lấy danh sách các tệp .flac và sắp xếp theo tên [cite: 20, 21, 25]
                audio_files = sorted([f for f in os.listdir(chapter_path) if f.endswith('.flac')])

                # Chỉ lấy tối đa 10 tệp đầu tiên
                files_to_merge = audio_files[:10]

                if files_to_merge:
                    merged_audio = AudioSegment.empty()

                    # Gộp các tệp âm thanh lại
                    for audio_file in files_to_merge:
                        file_path = os.path.join(chapter_path, audio_file)
                        audio_segment = AudioSegment.from_file(file_path, format="flac")
                        merged_audio += audio_segment

                    # Tạo tên tệp mới dựa trên speaker ID và chapter ID [cite: 24]
                    output_filename = f"{speaker_id}_{chapter_id}_first10.flac"
                    output_path = os.path.join(output_base_dir, output_filename)

                    # Xuất tệp đã gộp
                    merged_audio.export(output_path, format="flac")
                    print(f"Đã gộp và lưu: {output_filename}")

print("Hoàn tất!")