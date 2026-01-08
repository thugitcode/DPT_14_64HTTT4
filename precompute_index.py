import pickle
from src.search import build_index

def main():
    dataset_dir = "static/dataset/archive/caltech-101"
    index = build_index(dataset_dir)

    with open("features_index.pkl", "wb") as f:
        pickle.dump(index, f)

    print(f"Saved index with {len(index)} images")

if __name__ == "__main__":
    main()
