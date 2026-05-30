import os
import psycopg2
import whisper

DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "speech",
    "user": "postgres",
    "password": "123456"
}

def main():
    print("\n[1/4] Loading Whisper Speech-to-Text model...")
    # Using 'base' model for a good balance between speed and accuracy
    try:
        stt_model = whisper.load_model("base")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return

    print("[2/4] Connecting to PostgreSQL Database...")
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    # Replace this path with the audio file you want to query
    input_file = r"D:\College\mmds\merged_all_test.flac"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"\n[3/4] Processing input audio: '{os.path.basename(input_file)}'")
    print("      Listening and transcribing audio to text...")

    # Transcribe the audio file into a text string
    transcribe_result = stt_model.transcribe(input_file, language="en", fp16=False)
    input_text = transcribe_result["text"].lower()

    # Tokenize the text and extract unique words using set()
    # Converted to a tuple so it can be passed safely into the psycopg2 IN clause
    input_words = tuple(set(input_text.split()))

    # print(f"      -> Extracted Text: '{input_text}'")
    # print(f"      -> Unique Keywords Found: {len(input_words)} words")

    print("\n[4/4] Querying Inverted Index for matching content...")

    if input_words:
        query = """
            SELECT a.fileName, SUM(i.tf_idfScore), a.filePath as match_score
            FROM InvertedFile i
            JOIN Keyword k ON i.keywordId = k.keywordId
            JOIN Audio a ON i.audioId = a.audioId
            WHERE k.word IN %s
            GROUP BY a.audioId, a.fileName
            ORDER BY match_score DESC
            LIMIT 3;
        """
        try:
            cursor.execute(query, (input_words,))
            top_3_results = cursor.fetchall()

            input_path = os.path.abspath(input_file).replace('\\', '/')
            print("\n" + "=" * 50)
            print(f"TOP 3 CONTENT MATCHES FOR {os.path.basename(input_file)}: file:///{input_path}")
            print("=" * 50)

            if top_3_results:
                for rank, (file_name, score, file_path) in enumerate(top_3_results, 1):
                    abs_path = os.path.abspath(file_path).replace('\\', '/')
                    print(f" Rank {rank}: {file_name}: file:///{abs_path}")
                    print(f"   -> Similarity Score: {score:.4f}\n")
            else:
                print("No matching content found in the database for these keywords.")

        except Exception as e:
            print(f"Query execution failed: {e}")
            conn.rollback()
    else:
        print("No words were recognized in the input audio to perform a query.")

    # Clean up database connections to prevent memory leaks
    cursor.close()
    conn.close()
    # print("Database connection closed. System exited.")

if __name__ == "__main__":
    main()