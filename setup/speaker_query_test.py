import numpy as np
import pickle
import os
import ast
import pandas as pd

# --- Map the 16 dimensions to human-readable names ---
# Indices 0-12 are MFCCs, 13 is Silence, 14 is Harmonicity, 15 is Centroid
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

class KDNode:
    def __init__(self, point, audio_id, left=None, right=None, axis=0):
        self.point = point
        self.audio_id = audio_id
        self.left = left
        self.right = right
        self.axis = axis

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

def search_kdtree(root, target, k=5):
    best_nodes = [] 

    def search(node):
        if node is None: return

        dist = euclidean_distance(target, node.point)
        best_nodes.append((dist, node))
        best_nodes.sort(key=lambda x: x[0])
        if len(best_nodes) > k:
            best_nodes.pop()

        axis = node.axis
        diff = target[axis] - node.point[axis]
        close_branch = node.left if diff < 0 else node.right
        far_branch = node.right if diff < 0 else node.left

        search(close_branch)

        if len(best_nodes) < k or abs(diff) < best_nodes[-1][0]:
            search(far_branch)

    search(root)
    return best_nodes

def main():
    model_file = "../Hoang/local_kdtree_model.pkl"
    excel_file = "../audio_features_report.xlsx"

    if not os.path.exists(model_file) or not os.path.exists(excel_file):
        print("Error: Run extraction and tree building scripts first.")
        return

    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    kd_tree_root = model_data["tree_root"]
    audio_dict = model_data["audio_dict"]

    df = pd.read_excel(excel_file)
    
    # Select a target file to search for
    test_index = 100
    query_row = df.iloc[test_index] 
    query_filename = query_row['File_Name']
    query_vector = np.array(ast.literal_eval(query_row['MFCC_Mean_Vector']))

    print(f"\n{'='*80}")
    print(f"SEARCHING KD-TREE FOR: {query_filename}")
    print(f"{'='*80}\n")
    
    top_results = search_kdtree(kd_tree_root, query_vector, k=3)
    
    for rank, (dist, node) in enumerate(top_results, 1):
        f_name = audio_dict[node.audio_id]
        
        # 1. Calculate Similarity Percentage
        similarity_pct = max(0.0, (1 - (dist ** 2) / 2)) * 100
        
        # 2. Analyze the "Why" (Feature Importance)
        # Find the absolute difference between the query vector and this result's vector
        feature_diffs = np.abs(query_vector - node.point)
        
        # Get the indices of the 3 SMALLEST differences (the features that matched the closest)
        closest_indices = np.argsort(feature_diffs)[:3]
        driving_features = [FEATURE_NAMES[i] for i in closest_indices]
        
        # 3. Print the formatted result
        print(f"[{rank}] {f_name}")
        print(f"    ├─ Similarity: {similarity_pct:.2f}%  (Distance: {dist:.4f})")
        print(f"    └─ Strongest Matches: {driving_features[0]}, {driving_features[1]}, {driving_features[2]}\n")

if __name__ == "__main__":
    main()