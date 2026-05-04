#!/usr/bin/env python3
"""
Extract threads from the News feed with depth >= 4.

Optimized Strategy: 
1. Find candidate thread roots from News.jsonl
2. Get participants from threads.txt
3. Batch scan relevant user files ONCE to collect all posts
4. Build trees and filter by depth

Usage:
    python extract_news_threads.py --output threads_output.jsonl --num-threads 500 --min-depth 4
"""

import json
import os
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from tqdm import tqdm
import concurrent.futures


# Paths
DATA_DIR = Path("/shared/agamg2/projects/agentic-safety/data")
NEWS_FEED = DATA_DIR / "feed_posts" / "News.jsonl"
THREADS_FILE = DATA_DIR / "graphs" / "threads.txt"
USER_POSTS_DIR = DATA_DIR / "user_posts"
NEWS_LIKES = DATA_DIR / "feed_posts_likes" / "News.csv"


def parse_date(date_int: int) -> str:
    """Convert date integer (YYYYMMDDHHmm) to ISO format string."""
    try:
        date_str = str(date_int)
        if len(date_str) == 12:
            dt = datetime.strptime(date_str, "%Y%m%d%H%M")
            return dt.isoformat()
        elif len(date_str) == 8:
            dt = datetime.strptime(date_str, "%Y%m%d")
            return dt.isoformat()
    except:
        pass
    return str(date_int)


def load_news_root_posts() -> Dict[int, dict]:
    """Load root posts from News.jsonl that have replies."""
    print("Step 1: Loading News feed root posts with replies...")
    root_posts = {}
    seen = set()
    
    with open(NEWS_FEED) as f:
        for line in f:
            post = json.loads(line.strip())
            post_id = post['post_id']
            
            # Skip duplicates
            if post_id in seen:
                continue
            seen.add(post_id)
            
            # Only keep root posts (reply_to is None) with replies
            if post.get('reply_to') is None and post.get('reply_count', 0) > 0:
                root_posts[post_id] = post
    
    # Sort by reply_count to prioritize active threads
    sorted_posts = dict(sorted(
        root_posts.items(), 
        key=lambda x: x[1].get('reply_count', 0), 
        reverse=True
    ))
    
    print(f"  Found {len(sorted_posts)} root posts with replies")
    return sorted_posts


def load_thread_participants(target_thread_ids: Set[int]) -> Dict[int, Set[int]]:
    """Load participants only for specific thread IDs."""
    print("Step 2: Loading thread participants from threads.txt...")
    threads = {}
    
    with open(THREADS_FILE) as f:
        for line in tqdm(f, desc="Scanning threads.txt", total=19486141):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                thread_root_id = int(parts[0])
                if thread_root_id in target_thread_ids:
                    participants_str = parts[2]
                    participants = set(int(p) for p in participants_str.split(',') if p)
                    threads[thread_root_id] = participants
    
    print(f"  Found participants for {len(threads)} threads")
    return threads


def load_posts_for_users(user_ids: Set[int], target_thread_roots: Set[int]) -> Dict[int, List[dict]]:
    """Load all posts from specified users that belong to target threads."""
    print(f"Step 3: Loading posts from {len(user_ids)} user files...")
    
    # Group posts by thread_root
    thread_posts = defaultdict(list)
    posts_found = 0
    files_with_errors = 0
    
    for user_id in tqdm(user_ids, desc="Scanning user files"):
        user_file = USER_POSTS_DIR / f"{user_id}.jsonl"
        if not user_file.exists():
            continue
        
        try:
            # Use errors='replace' to handle non-UTF-8 characters
            with open(user_file, encoding='utf-8', errors='replace') as f:
                for line in f:
                    try:
                        post = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    post_id = post['post_id']
                    thread_root = post.get('thread_root')
                    
                    # Check if this post is a root we care about
                    if post_id in target_thread_roots and thread_root is None:
                        thread_posts[post_id].append(post)
                        posts_found += 1
                    # Check if this post belongs to a thread we care about
                    elif thread_root in target_thread_roots:
                        thread_posts[thread_root].append(post)
                        posts_found += 1
        except Exception as e:
            files_with_errors += 1
            continue
    
    print(f"  Found {posts_found} posts across {len(thread_posts)} threads")
    if files_with_errors > 0:
        print(f"  Warning: {files_with_errors} files had read errors (skipped)")
    return dict(thread_posts)


def calculate_thread_depth(posts: List[dict]) -> int:
    """Calculate the maximum depth of the thread tree."""
    if not posts:
        return 0
    
    by_id = {p['post_id']: p for p in posts}
    
    # Find root
    root_id = None
    for p in posts:
        if p.get('reply_to') is None:
            root_id = p['post_id']
            break
    
    if root_id is None:
        return 1
    
    # Build children mapping
    children = defaultdict(list)
    for p in posts:
        parent_id = p.get('reply_to')
        if parent_id and parent_id in by_id:
            children[parent_id].append(p['post_id'])
    
    # DFS to find max depth
    def get_depth(post_id: int) -> int:
        if post_id not in children or not children[post_id]:
            return 1
        return 1 + max(get_depth(child_id) for child_id in children[post_id])
    
    return get_depth(root_id)


def build_thread_tree(posts: List[dict]) -> Optional[dict]:
    """Build a hierarchical tree structure from flat list of posts."""
    if not posts:
        return None
    
    by_id = {p['post_id']: p.copy() for p in posts}
    
    # Find root
    root = None
    for p in posts:
        if p.get('reply_to') is None:
            root = by_id[p['post_id']]
            break
    
    if not root:
        # Find post whose reply_to is not in our collection
        for p in posts:
            if p.get('reply_to') not in by_id:
                root = by_id[p['post_id']]
                break
    
    if not root:
        # Fallback: earliest post
        earliest = min(posts, key=lambda x: x.get('date', float('inf')))
        root = by_id[earliest['post_id']]
    
    # Initialize children lists
    for post_id in by_id:
        by_id[post_id]['children'] = []
    
    # Build parent-child relationships
    orphans = []
    for post_id, post in by_id.items():
        parent_id = post.get('reply_to')
        if parent_id and parent_id in by_id:
            by_id[parent_id]['children'].append(post)
        elif parent_id and parent_id not in by_id and post_id != root['post_id']:
            orphans.append(post)
    
    # Attach orphans to root
    for orphan in orphans:
        root['children'].append(orphan)
    
    # Sort children by date
    def sort_children(node):
        node['children'].sort(key=lambda x: x.get('date', 0))
        for child in node['children']:
            sort_children(child)
    
    sort_children(root)
    return root


def format_post_for_output(post: dict) -> dict:
    """Format a post for clean output."""
    return {
        "post_id": post.get('post_id'),
        "user_id": post.get('user_id'),
        "text": post.get('text', ''),
        "date": parse_date(post.get('date', 0)),
        "date_raw": post.get('date'),
        "langs": post.get('langs', []),
        "like_count": post.get('like_count', 0),
        "reply_count": post.get('reply_count', 0),
        "repost_count": post.get('repost_count', 0),
        "instance": post.get('instance', ''),
        "labels": post.get('labels'),
        "sentiment_label": post.get('sent_label'),
        "sentiment_score": post.get('sent_score'),
        "children": [format_post_for_output(c) for c in post.get('children', [])]
    }


def count_posts_in_tree(node: dict) -> int:
    """Count total posts in a tree."""
    count = 1
    for child in node.get('children', []):
        count += count_posts_in_tree(child)
    return count


def get_tree_depth(node: dict, current_depth: int = 1) -> int:
    """Get maximum depth of tree."""
    if not node.get('children'):
        return current_depth
    return max(get_tree_depth(child, current_depth + 1) for child in node['children'])


def get_all_user_ids(node: dict) -> Set[int]:
    """Get all unique user_ids in a tree."""
    user_ids = {node.get('user_id')}
    for child in node.get('children', []):
        user_ids.update(get_all_user_ids(child))
    return user_ids


def sum_engagement(node: dict) -> dict:
    """Sum up all engagement metrics in tree."""
    totals = {
        'likes': node.get('like_count', 0) or 0,
        'replies': node.get('reply_count', 0) or 0,
        'reposts': node.get('repost_count', 0) or 0
    }
    for child in node.get('children', []):
        child_totals = sum_engagement(child)
        totals['likes'] += child_totals['likes']
        totals['replies'] += child_totals['replies']
        totals['reposts'] += child_totals['reposts']
    return totals


def extract_threads(
    num_threads: int = 500,
    min_depth: int = 4,
    output_file: str = "news_threads.jsonl"
) -> List[dict]:
    """Main extraction function - optimized version."""
    
    # Step 1: Load root posts from News feed
    root_posts = load_news_root_posts()
    target_thread_ids = set(root_posts.keys())
    
    # Step 2: Load participants for these threads
    thread_participants = load_thread_participants(target_thread_ids)
    
    # Filter to threads we have participants for
    valid_thread_ids = set(thread_participants.keys())
    print(f"\nThreads with participant data: {len(valid_thread_ids)}")
    
    # Collect all user IDs we need to scan
    all_user_ids = set()
    for participants in thread_participants.values():
        all_user_ids.update(participants)
    print(f"Total unique users to scan: {len(all_user_ids)}")
    
    # Step 3: Load all posts from these users for target threads
    thread_posts = load_posts_for_users(all_user_ids, valid_thread_ids)
    
    # Step 4: Build trees and filter by depth
    print(f"\nStep 4: Building thread trees and filtering by depth >= {min_depth}...")
    
    extracted_threads = []
    skipped_shallow = 0
    skipped_empty = 0
    
    # Sort thread_ids by the original reply_count (most active first)
    sorted_thread_ids = sorted(
        valid_thread_ids,
        key=lambda tid: root_posts.get(tid, {}).get('reply_count', 0),
        reverse=True
    )
    
    for thread_root_id in tqdm(sorted_thread_ids, desc="Building trees"):
        posts = thread_posts.get(thread_root_id, [])
        
        if len(posts) < min_depth:
            skipped_shallow += 1
            continue
        
        # Quick depth check before building full tree
        depth = calculate_thread_depth(posts)
        if depth < min_depth:
            skipped_shallow += 1
            continue
        
        # Build the tree
        tree = build_thread_tree(posts)
        if not tree:
            skipped_empty += 1
            continue
        
        # Verify depth in built tree
        actual_depth = get_tree_depth(tree)
        if actual_depth < min_depth:
            skipped_shallow += 1
            continue
        
        # Format for output
        formatted_tree = format_post_for_output(tree)
        
        # Calculate metadata
        total_posts = count_posts_in_tree(formatted_tree)
        unique_users = get_all_user_ids(tree)
        engagement = sum_engagement(tree)
        
        thread_data = {
            "thread_id": thread_root_id,
            "source_feed": "News",
            "depth": actual_depth,
            "total_posts": total_posts,
            "unique_participants": len(unique_users),
            "participant_ids": list(unique_users),
            "total_likes": engagement['likes'],
            "total_reposts": engagement['reposts'],
            "root_post_date": formatted_tree.get('date'),
            "root": formatted_tree
        }
        
        extracted_threads.append(thread_data)
        
        if len(extracted_threads) >= num_threads:
            break
    
    print(f"\n=== Extraction Summary ===")
    print(f"  Threads with depth >= {min_depth}: {len(extracted_threads)}")
    print(f"  Skipped (too shallow): {skipped_shallow}")
    print(f"  Skipped (empty/invalid): {skipped_empty}")
    
    # Save to file
    output_path = Path(output_file)
    print(f"\nSaving {len(extracted_threads)} threads to {output_path}...")
    
    with open(output_path, 'w') as f:
        for thread in extracted_threads:
            f.write(json.dumps(thread, ensure_ascii=False) + '\n')
    
    # Also save a summary JSON
    summary_file = output_path.with_suffix('.summary.json')
    summary = {
        "source_feed": "News",
        "extraction_date": datetime.now().isoformat(),
        "total_threads": len(extracted_threads),
        "min_depth_filter": min_depth,
        "depth_distribution": {},
        "avg_posts_per_thread": 0,
        "avg_participants_per_thread": 0,
        "total_posts_extracted": 0,
        "total_unique_users": 0
    }
    
    all_users = set()
    for thread in extracted_threads:
        depth = thread['depth']
        summary['depth_distribution'][str(depth)] = summary['depth_distribution'].get(str(depth), 0) + 1
        summary['total_posts_extracted'] += thread['total_posts']
        all_users.update(thread['participant_ids'])
    
    if extracted_threads:
        summary['avg_posts_per_thread'] = round(
            summary['total_posts_extracted'] / len(extracted_threads), 2
        )
        summary['avg_participants_per_thread'] = round(
            sum(t['unique_participants'] for t in extracted_threads) / len(extracted_threads), 2
        )
    
    summary['total_unique_users'] = len(all_users)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {summary_file}")
    
    # Also save a human-readable sample
    sample_file = output_path.with_suffix('.sample.json')
    if extracted_threads:
        with open(sample_file, 'w') as f:
            json.dump(extracted_threads[0], f, indent=2, ensure_ascii=False)
        print(f"Saved sample thread to {sample_file}")
    
    return extracted_threads


def main():
    parser = argparse.ArgumentParser(description="Extract threads from News feed")
    parser.add_argument(
        '--output', '-o',
        default='/shared/agamg2/projects/red_teaming_agent/data/news_threads.jsonl',
        help='Output file path'
    )
    parser.add_argument(
        '--num-threads', '-n',
        type=int,
        default=500,
        help='Number of threads to extract'
    )
    parser.add_argument(
        '--min-depth', '-d',
        type=int,
        default=4,
        help='Minimum thread depth'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_threads(
        num_threads=args.num_threads,
        min_depth=args.min_depth,
        output_file=args.output
    )


if __name__ == "__main__":
    main()



