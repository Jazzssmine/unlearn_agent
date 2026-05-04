import json
import csv
import glob
import os
import argparse

def process_single_thread(thread_data):
    """
    Traverses a single nested thread object to find chains of:
    Root (OP) -> Child (Author 1) -> Grandchild (Author 2) -> [Optional OP Return]
    """
    results = []
    
    # Access the root post
    root = thread_data.get("root")
    if not root:
        return []

    # 1. OP (Root)
    op_id = str(root.get("user_id"))
    op_text = root.get("text", "")
    op_post_id = str(root.get("post_id"))

    # Iterate through immediate replies (Author 1 candidates)
    for child in root.get("children", []):
        a1_id = str(child.get("user_id"))
        a1_text = child.get("text", "")
        a1_post_id = str(child.get("post_id"))

        # Iterate through replies to Author 1 (Author 2 candidates)
        for grandchild in child.get("children", []):
            a2_id = str(grandchild.get("user_id"))
            a2_text = grandchild.get("text", "")
            a2_post_id = str(grandchild.get("post_id"))

            # We found a valid 3-step chain
            # Generate a unique ID for this specific branch
            chain_id = f"{op_post_id}_{a1_post_id}_{a2_post_id}"
            
            # Build the sequence rows
            chain_rows = [
                # Sequence 1: OP
                {"thread_id": chain_id, "sequence": 1, "author_id": op_id, "content": op_text},
                # Sequence 2: Author 1
                {"thread_id": chain_id, "sequence": 2, "author_id": a1_id, "content": a1_text},
                # Sequence 3: Author 2
                {"thread_id": chain_id, "sequence": 3, "author_id": a2_id, "content": a2_text}
            ]

            # Optional Sequence 4: Check if OP replied back to Author 2
            # We look into grandchild's children for the OP's user_id
            for great_grandchild in grandchild.get("children", []):
                if str(great_grandchild.get("user_id")) == op_id:
                    chain_rows.append({
                        "thread_id": chain_id,
                        "sequence": 4,
                        "author_id": op_id,
                        "content": great_grandchild.get("text", "")
                    })
                    # We take the first reply from OP and stop checking
                    break 

            results.extend(chain_rows)

    return results

def load_and_process_files(input_dir, output_file):
    all_rows = []
    
    # Support both .json and .jsonl files
    files = glob.glob(os.path.join(input_dir, "*.json")) + glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    print(f"Found {len(files)} files to process in {input_dir}")

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Handle JSONL (one object per line) or standard JSON (one object per file or list of objects)
            content = f.read().strip()
            if not content:
                continue

            try:
                # Try loading as a single JSON object or list of objects
                data = json.loads(content)
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
            except json.JSONDecodeError:
                # If that fails, try processing line by line (JSONL)
                f.seek(0)
                items = []
                for line in f:
                    if line.strip():
                        try:
                            items.append(json.loads(line))
                        except:
                            continue

            # Process each thread object found
            for thread_obj in items:
                # Basic validation to ensure it matches the schema
                if "root" in thread_obj:
                    rows = process_single_thread(thread_obj)
                    all_rows.extend(rows)

    # Save to CSV
    if all_rows:
        keys = ["thread_id", "sequence", "author_id", "content"]
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        
        # Calculate unique threads (divide by typical length 3 or 4 to estimate)
        unique_threads = len(set(r['thread_id'] for r in all_rows))
        print(f"Successfully extracted {unique_threads} conversation chains.")
        print(f"Data saved to {output_file}")
    else:
        print("No valid conversation chains (Root -> Child -> Grandchild) found.")

def main():
    parser = argparse.ArgumentParser(description="Convert nested JSON threads to simulation CSV")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing thread JSON files")
    parser.add_argument("--output_file", type=str, default="threads_data.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    load_and_process_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()

"""
python src/thread_extraction/process_threads.py --input_dir ./data/ --output_file ./data/threads_data.csv
"""