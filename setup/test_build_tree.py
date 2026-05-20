import pandas as pd
import numpy as np
import ast
import pickle
import os

class KDNode:
    def __init__(self, point, audio_id, left=None, right=None, axis=0):
        self.point = point
        self.audio_id = audio_id
        self.left = left
        self.right = right
        self.axis = axis

def build_kdtree(points, ids, depth=0):
    if len(points) == 0:
        return None
    k = len(points[0])
    axis = depth % k
    combined = list(zip(points, ids))
    combined.sort(key=lambda x: x[0][axis])
    
    median_idx = len(combined) // 2
    median_point, median_id = combined[median_idx]

    left_branch = build_kdtree(
        [x[0] for x in combined[:median_idx]],
        [x[1] for x in combined[:median_idx]], depth + 1)
    right_branch = build_kdtree(
        [x[0] for x in combined[median_idx + 1:]],
        [x[1] for x in combined[median_idx + 1:]], depth + 1)

    return KDNode(median_point, median_id, left_branch, right_branch, axis)

def main():
    excel_file = "../audio_features_report.xlsx"
    output_file = "../Hoang/local_kdtree_model.pkl"

    print("1. Loading Data from Excel...")
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found.")
        return

    df = pd.read_excel(excel_file)
    points = []
    ids = []
    audio_dict = {}

    print("2. Formatting Vectors...")
    for idx, row in df.iterrows():
        audio_id = idx + 1 
        audio_dict[audio_id] = row['File_Name']
        vector = np.array(ast.literal_eval(row['MFCC_Mean_Vector']))
        
        points.append(vector)
        ids.append(audio_id)

    print("3. Building KD-Tree...")
    kd_tree_root = build_kdtree(points, ids)

    model_data = {
        "tree_root": kd_tree_root,
        "audio_dict": audio_dict
    }

    print(f"4. Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Done! Tree is ready for searching.")

if __name__ == "__main__":
    main()