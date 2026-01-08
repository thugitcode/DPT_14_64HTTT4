import argparse
import os
from src.search import build_index, search

def main():
    parser = argparse.ArgumentParser(description="Image similarity search (LBP + Gabor + Cosine)")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--query", required=True, help="Path to query image")
    parser.add_argument("--topk", type=int, default=5, help="Top K results to return")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")
    if not os.path.isfile(args.query):
        raise FileNotFoundError(f"Query image not found: {args.query}")

    print("Building index...")
    index = build_index(args.dataset)
    print(f"Indexed {len(index)} images")

    print("Searching...")
    results = search(args.query, index, topk=args.topk)
    for fname, score in results:
        print(f"{fname}\t{score:.4f}")

if __name__ == "__main__":
    main()
