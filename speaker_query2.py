import os
import numpy as np
import librosa
import pickle
import warnings
import psycopg2
import ast

warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160

# Database connection parameters
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "speech"
DB_USER = "postgres"
DB_PASS = "123456"

FEATURE_NAMES = [
    "MFCC 1 (Overall Energy Envelope)",
    "MFCC 2 (Broad Vocal Tract Shape)",
    "MFCC 3 (Tongue Position/Vowels)",
    "MFCC 4", "MFCC 5", "MFCC 6", "MFCC 7", "MFCC 8",
    "MFCC 9", "MFCC 10", "MFCC 11", "MFCC 12", "MFCC 13",
    "Silence Ratio (Speech Pacing)",
    "Harmonicity (Resonance vs. Raspiness)",
    "Spectral Centroid (Voice Brightness/Tone)"
]


def extract_and_normalize(file_path, means, stds):
    """
    Extracts 16-D audio features and applies Z-score standardization
    followed by L2 normalization.
    """
    try:
        y_raw, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        rms_raw = librosa.feature.rms(y=y_raw, frame_length=N_FFT, hop_length=HOP_LENGTH)
        silence_ratio = np.sum(rms_raw < 0.01) / float(rms_raw.shape[1])

        intervals = librosa.effects.split(y_raw, top_db=20)
        if len(intervals) == 0: return None
        y_clean = np.concatenate([y_raw[start:end] for start, end in intervals])

        mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfccs, axis=1)

        centroid = librosa.feature.spectral_centroid(y=y_clean, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        centroid_mean = np.mean(centroid)

        y_harmonic, _ = librosa.effects.hpss(y_clean)
        harmonicity = np.mean(librosa.feature.rms(y=y_harmonic, frame_length=N_FFT, hop_length=HOP_LENGTH))

        raw_vector = np.concatenate((mfcc_mean, [silence_ratio, harmonicity, centroid_mean]))

        # Apply Z-score standardization
        standardized = (raw_vector - means) / stds

        # Apply L2 Normalization
        norm = np.linalg.norm(standardized)
        l2_vector = standardized if norm == 0 else standardized / norm

        return l2_vector

    except Exception as e:
        print(f"Error during extraction: {e}")
        return None


def search_database(query_vector, k=3):
    """
    Connects to PostgreSQL and uses pgvector's L2 distance operator (<->)
    to find the top k closest matches.
    """
    try:
        # 1. Establish database connection
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cursor = conn.cursor()

        # Format vector as a string array for PostgreSQL pgvector compatibility
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        # 2. Execute vector search query
        # Using <-> for Euclidean distance natively supported by pgvector
        query = """
            SELECT fileName, speakerFeature, (speakerFeature <-> %s::vector) AS distance, filePath
            FROM Audio
            ORDER BY distance ASC
            LIMIT %s;
        """
        cursor.execute(query, (vector_str, k))
        results = cursor.fetchall()

        # 3. Close connections
        cursor.close()
        conn.close()

        return results

    except Exception as e:
        print(f"Database Error: {e}")
        return []


def main():
    scaler_file = "tree/scaler_params.pkl"

    if not os.path.exists(scaler_file):
        print("Error: Missing scaler_params.pkl file!")
        return

    print("[1/3] Loading global scaler parameters into RAM...")
    with open(scaler_file, "rb") as f:
        scaler_data = pickle.load(f)

    GLOBAL_MEANS = scaler_data["means"]
    GLOBAL_STDS = scaler_data["stds"]

    # Target audio file for the query
    input_str = input("\nEnter the audio file: ")
    query_audio_path = input_str.strip().strip("'").strip('"')

    # Remove surrounding quotes if dragged-and-dropped into terminal
    if query_audio_path.startswith(('"', "'")) and query_audio_path.endswith(('"', "'")):
        query_audio_path = query_audio_path[1:-1]

    if not os.path.exists(query_audio_path):
        print("Error: Audio file does not exist.")
        return

    print(f"[2/3] Analyzing voice and extracting features...")
    query_vector = extract_and_normalize(query_audio_path, GLOBAL_MEANS, GLOBAL_STDS)

    if query_vector is None:
        return

    query_url = query_audio_path.replace('\\', '/')
    print(f"\n{'=' * 80}")
    print(f"SEARCHING DATABASE FOR: {os.path.basename(query_audio_path)}: file:///{query_url}")
    print(f"{'=' * 80}\n")

    print("[3/3] Querying PostgreSQL pgvector...")
    top_results = search_database(query_vector, k=3)

    if not top_results:
        print("No results found. Please check your database connection and data.")
        return

    # Process and display results
    for rank, (f_name, db_vector_str, dist, file_path) in enumerate(top_results, 1):

        # Convert pgvector string format '[v1, v2, ...]' back to a numpy array for difference calculation
        if isinstance(db_vector_str, str):
            db_vector = np.array([float(x) for x in db_vector_str.strip('[]').split(',')])
        else:
            # Fallback if psycopg2 automatically maps the vector array
            db_vector = np.array(db_vector_str)

        # Calculate similarity percentage based on Euclidean distance
        similarity_pct = max(0.0, (1 - (dist ** 2) / 2)) * 100

        # Calculate absolute difference to find driving features
        feature_diffs = np.abs(query_vector - db_vector)
        closest_indices = np.argsort(feature_diffs)[:3]
        driving_features = [FEATURE_NAMES[i] for i in closest_indices]

        abs_path = os.path.abspath(file_path).replace('\\', '/')
        print(f"[{rank}] {f_name}: file:///{abs_path}")
        print(f"    ├─ Similarity: {similarity_pct:.2f}%  (Distance: {dist:.4f})")
        print(f"    └─ Strongest Matches: {driving_features[0]}, {driving_features[1]}, {driving_features[2]}\n")


if __name__ == "__main__":
    main()