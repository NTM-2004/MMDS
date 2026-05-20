import pandas as pd
import numpy as np
import ast 

DATABASE_FILE = "../audio_features_report.xlsx"

def load_database(file_path):
    """Loads the standardized 17-D vectors from the Excel file."""
    try:
        df = pd.read_excel(file_path)
        database = {}
        for _, row in df.iterrows():
            filename = row['File_Name']
            # Convert the string representation of the list back into a Numpy array
            vector = np.array(ast.literal_eval(row['MFCC_Mean_Vector']))
            database[filename] = vector 
        return database
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Run the extraction script first.")
        exit()
    except Exception as e:
        print(f"Error loading database: {e}")
        exit()

def euclidean_distance(p1, p2):
    """
    Calculates the Euclidean distance. 
    Since the data is already standardized, all 17 features have equal weight.
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))

def search_similar(query_filename, db, top_n=10):
    if query_filename not in db:
        print(f"Error: '{query_filename}' not found in the database.")
        return
    
    query_vector = db[query_filename]
    results = []
    
    for db_filename, db_vector in db.items():
        if db_filename == query_filename:
            continue
            
        dist = euclidean_distance(query_vector, db_vector)
        
        # --- The Similarity Curve ---
        # Using the Smoothing Constant (C=10) we discussed to give realistic 
        # percentages for standardized data.
        C = 10 
        similarity_pct = (C / (C + dist)) * 100
        
        results.append((db_filename, dist, similarity_pct))
        
    # Sort by Distance (Ascending: lower distance = higher similarity)
    results.sort(key=lambda x: x[1])
    
    print(f"\n--- Searching Voice Matches for: {query_filename} ---")
    print(f"{'Rank':<5} | {'Filename':<45} | {'Distance':<10} | {'Similarity'}")
    print("-" * 85)
    
    for i in range(min(top_n, len(results))):
        name, dist, pct = results[i]
        print(f"{i+1:<5} | {name:<45} | {dist:<10.4f} | {pct:.2f}%")

def main():
    print("Loading 17-D Voice Database...")
    database = load_database(DATABASE_FILE)
    
    if database:
        files_list = list(database.keys())
        
        # Change this index to test different audio files in your folder
        test_query = files_list[0] 
        search_similar(test_query, database, top_n=10)

if __name__ == "__main__":
    main()