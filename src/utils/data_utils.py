"""
Data utilities for the toxic agent project.
get formalized threads from the data.
"""
import json
import csv
import glob
import os
import argparse
from collections import defaultdict

def load_posts_from_files(data_path):
    """
    Reads all .jsonl files in the directory and loads them into a dictionary.
    Returns:
        posts_map: dict {post_id: post_data_dict}
        children_map: dict {parent_post_id: [list_of_child_post_ids]}
    """
    posts_map = {}
    children_map = defaultdict(list)
    
    # Match all .jsonl files in the provided path
    file_pattern = os.path.join(data_path, "*.jsonl")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No .jsonl files found in {data_path}")
        return {}, {}

    print(f"Loading data from {len(files)} files...")
    
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Clean up line and parse JSON
                    line = line.strip()
                    if not line:
                        continue
                    post = json.loads(line)
                    
                    pid = post.get('post_id')
                    if not pid:
                        continue
                        
                    # Store post data (deduplicate by ID)
                    if pid not in posts_map:
                        posts_map[pid] = post
                        
                        # Build adjacency relationship
                        # reply_to indicates the parent. 
                        # If reply_to is null, it might be a root.
                        parent_id = post.get('reply_to')
                        if parent_id:
                            children_map[parent_id].append(pid)
                            
                except json.JSONDecodeError:
                    continue
                    
    print(f"Total unique posts loaded: {len(posts_map)}")
    return posts_map, children_map

def extract_conversations(posts_map, children_map):
    """
    Finds chains of length 3: OP -> Reply1 -> Reply2
    """
    threads_data = []
    
    # 1. Identify Potential Roots
    # A root is usually defined as having no reply_to, OR its thread_root equals its post_id
    # We iterate over all posts to find valid starting points for our chains
    
    print("reconstructing conversation threads...")
    
    # We track processed chains to avoid duplicates
    processed_chains = set()

    for pid, post in posts_map.items():
        # Check if this is a Root (OP)
        # Criteria: No reply_to, or explicitly marked as root
        reply_to = post.get('reply_to')
        thread_root = post.get('thread_root')
        
        is_root = (reply_to is None) or (thread_root == pid)
        
        if is_root:
            op_id = pid
            
            # LEVEL 1: Look for Author 1 (Direct replies to OP)
            replies_to_op = children_map.get(op_id, [])
            
            for reply1_id in replies_to_op:
                reply1 = posts_map[reply1_id]
                
                # LEVEL 2: Look for Author 2 (Replies to Author 1)
                replies_to_r1 = children_map.get(reply1_id, [])
                
                for reply2_id in replies_to_r1:
                    reply2 = posts_map[reply2_id]
                    
                    # We have found a chain: OP -> Reply1 -> Reply2
                    # Create a unique Thread ID for this specific interaction chain
                    # (We use the root ID + suffix to distinguish distinct branches if needed)
                    chain_id = f"{op_id}_{reply1_id}_{reply2_id}"
                    
                    if chain_id in processed_chains:
                        continue
                        
                    # Add Sequence 1: OP
                    threads_data.append({
                        "thread_id": chain_id,
                        "sequence": 1,
                        "author_id": post.get('user_id'),
                        "content": post.get('text', "")
                    })
                    
                    # Add Sequence 2: Author 1
                    threads_data.append({
                        "thread_id": chain_id,
                        "sequence": 2,
                        "author_id": reply1.get('user_id'),
                        "content": reply1.get('text', "")
                    })
                    
                    # Add Sequence 3: Author 2
                    threads_data.append({
                        "thread_id": chain_id,
                        "sequence": 3,
                        "author_id": reply2.get('user_id'),
                        "content": reply2.get('text', "")
                    })
                    
                    # Add Sequence 4: OP (Closing the loop as per your diagram)
                    # "OP | Author_1 | Author_2 | OP"
                    # We check if the OP replied back to Author 2
                    replies_to_r2 = children_map.get(reply2_id, [])
                    op_closing_reply = None
                    
                    for reply3_id in replies_to_r2:
                        reply3 = posts_map[reply3_id]
                        if reply3.get('user_id') == post.get('user_id'):
                            op_closing_reply = reply3
                            break # Found the OP coming back
                    
                    if op_closing_reply:
                        threads_data.append({
                            "thread_id": chain_id,
                            "sequence": 4,
                            "author_id": op_closing_reply.get('user_id'),
                            "content": op_closing_reply.get('text', "")
                        })
                    
                    processed_chains.add(chain_id)

    return threads_data

def main():
    parser = argparse.ArgumentParser(description="Convert raw JSONL posts to conversation threads CSV")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing user .jsonl files")
    parser.add_argument("--output_file", type=str, default="threads_data.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    
    # 1. Load Data
    posts, relations = load_posts_from_files(args.input_dir)
    
    if not posts:
        return

    # 2. Build Threads
    data = extract_conversations(posts, relations)
    print(f"Found {len(data) // 3} valid conversation chains (depth >= 3).")
    
    # 3. Save to CSV
    if data:
        keys = ["thread_id", "sequence", "author_id", "content"]
        with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully saved to {args.output_file}")
    else:
        print("No conversations matching the criteria (OP -> A1 -> A2) were found.")

if __name__ == "__main__":
    main()