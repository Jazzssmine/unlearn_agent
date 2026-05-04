#!/usr/bin/env python3
"""
Thread Viewer - Explore extracted threads in a human-readable format.

Usage:
    python view_threads.py data/news_threads.jsonl                    # View summary
    python view_threads.py data/news_threads.jsonl --thread 0         # View first thread
    python view_threads.py data/news_threads.jsonl --thread 12345     # View thread by ID
    python view_threads.py data/news_threads.jsonl --search "keyword" # Search threads
    python view_threads.py data/news_threads.jsonl --stats            # Show statistics
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def load_threads(filepath: str) -> List[dict]:
    """Load threads from JSONL file."""
    threads = []
    with open(filepath) as f:
        for line in f:
            threads.append(json.loads(line.strip()))
    return threads


def truncate_text(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def format_tree_ascii(node: dict, prefix: str = "", is_last: bool = True, max_depth: int = 10) -> str:
    """Format a thread tree as ASCII art."""
    lines = []
    
    # Current node
    connector = "└── " if is_last else "├── "
    
    text_preview = truncate_text(node.get('text', '[no text]').replace('\n', ' '), 60)
    user_id = node.get('user_id', '?')
    likes = node.get('like_count', 0)
    date = node.get('date', '')[:10] if node.get('date') else ''
    
    node_str = f"[User {user_id}] {text_preview}"
    meta_str = f"  (❤️ {likes}, {date})"
    
    if prefix == "":
        lines.append(f"📝 {node_str}")
        lines.append(f"   {meta_str}")
    else:
        lines.append(f"{prefix}{connector}{node_str}")
        lines.append(f"{prefix}{'    ' if is_last else '│   '}{meta_str}")
    
    # Children
    children = node.get('children', [])
    if max_depth > 1:
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(format_tree_ascii(child, child_prefix, is_child_last, max_depth - 1))
    elif children:
        child_prefix = prefix + ("    " if is_last else "│   ")
        lines.append(f"{child_prefix}└── ... ({len(children)} more replies)")
    
    return "\n".join(lines)


def print_thread_summary(threads: List[dict], limit: int = 20):
    """Print a summary table of threads."""
    print(f"\n{'='*80}")
    print(f"THREAD SUMMARY ({len(threads)} threads)")
    print(f"{'='*80}")
    print(f"{'#':<5} {'Thread ID':<12} {'Depth':<6} {'Posts':<7} {'Users':<7} {'Likes':<8} {'Root Text Preview'}")
    print(f"{'-'*5} {'-'*12} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*35}")
    
    for i, thread in enumerate(threads[:limit]):
        thread_id = thread['thread_id']
        depth = thread['depth']
        posts = thread['total_posts']
        users = thread['unique_participants']
        likes = thread['total_likes']
        text = truncate_text(thread['root'].get('text', '')[:35].replace('\n', ' '), 35)
        
        print(f"{i:<5} {thread_id:<12} {depth:<6} {posts:<7} {users:<7} {likes:<8} {text}")
    
    if len(threads) > limit:
        print(f"\n... and {len(threads) - limit} more threads")


def print_thread_detail(thread: dict, max_depth: int = 10):
    """Print detailed view of a single thread."""
    print(f"\n{'='*80}")
    print(f"THREAD DETAIL: {thread['thread_id']}")
    print(f"{'='*80}")
    
    print(f"\n📊 Metadata:")
    print(f"   Thread ID:     {thread['thread_id']}")
    print(f"   Source Feed:   {thread['source_feed']}")
    print(f"   Depth:         {thread['depth']}")
    print(f"   Total Posts:   {thread['total_posts']}")
    print(f"   Participants:  {thread['unique_participants']}")
    print(f"   Total Likes:   {thread['total_likes']}")
    print(f"   Total Reposts: {thread['total_reposts']}")
    print(f"   Root Date:     {thread['root_post_date']}")
    
    print(f"\n📜 Conversation Tree:")
    print(f"{'-'*80}")
    print(format_tree_ascii(thread['root'], max_depth=max_depth))
    print(f"{'-'*80}")


def print_full_thread(thread: dict):
    """Print full text of all posts in thread (flattened)."""
    print(f"\n{'='*80}")
    print(f"FULL THREAD: {thread['thread_id']}")
    print(f"{'='*80}")
    
    def print_node(node: dict, depth: int = 0):
        indent = "  " * depth
        user = node.get('user_id', '?')
        date = node.get('date', '')
        text = node.get('text', '[no text]')
        likes = node.get('like_count', 0)
        
        print(f"\n{indent}┌─ User {user} | {date} | ❤️ {likes}")
        for line in text.split('\n'):
            print(f"{indent}│ {line}")
        print(f"{indent}└─")
        
        for child in node.get('children', []):
            print_node(child, depth + 1)
    
    print_node(thread['root'])


def search_threads(threads: List[dict], query: str) -> List[dict]:
    """Search threads for a keyword."""
    query_lower = query.lower()
    results = []
    
    def search_node(node: dict) -> bool:
        text = node.get('text', '').lower()
        if query_lower in text:
            return True
        for child in node.get('children', []):
            if search_node(child):
                return True
        return False
    
    for thread in threads:
        if search_node(thread['root']):
            results.append(thread)
    
    return results


def print_statistics(threads: List[dict]):
    """Print detailed statistics about the threads."""
    print(f"\n{'='*80}")
    print(f"THREAD STATISTICS")
    print(f"{'='*80}")
    
    if not threads:
        print("No threads to analyze.")
        return
    
    # Basic counts
    total_posts = sum(t['total_posts'] for t in threads)
    total_likes = sum(t['total_likes'] for t in threads)
    total_reposts = sum(t['total_reposts'] for t in threads)
    all_users = set()
    for t in threads:
        all_users.update(t['participant_ids'])
    
    print(f"\n📊 Overview:")
    print(f"   Total Threads:      {len(threads)}")
    print(f"   Total Posts:        {total_posts}")
    print(f"   Total Likes:        {total_likes}")
    print(f"   Total Reposts:      {total_reposts}")
    print(f"   Unique Users:       {len(all_users)}")
    
    # Depth distribution
    depths = [t['depth'] for t in threads]
    print(f"\n📏 Depth Distribution:")
    depth_counts = {}
    for d in depths:
        depth_counts[d] = depth_counts.get(d, 0) + 1
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        bar = "█" * (count // 2)
        print(f"   Depth {depth}: {count:>4} {bar}")
    
    # Averages
    print(f"\n📈 Averages:")
    print(f"   Avg Posts/Thread:        {total_posts / len(threads):.1f}")
    print(f"   Avg Participants/Thread: {sum(t['unique_participants'] for t in threads) / len(threads):.1f}")
    print(f"   Avg Likes/Thread:        {total_likes / len(threads):.1f}")
    print(f"   Avg Depth:               {sum(depths) / len(depths):.1f}")
    
    # Top threads
    print(f"\n🏆 Top Threads by Posts:")
    top_by_posts = sorted(threads, key=lambda t: t['total_posts'], reverse=True)[:5]
    for i, t in enumerate(top_by_posts, 1):
        text = truncate_text(t['root'].get('text', '')[:40], 40)
        print(f"   {i}. {t['total_posts']} posts - {text}")
    
    print(f"\n❤️ Top Threads by Likes:")
    top_by_likes = sorted(threads, key=lambda t: t['total_likes'], reverse=True)[:5]
    for i, t in enumerate(top_by_likes, 1):
        text = truncate_text(t['root'].get('text', '')[:40], 40)
        print(f"   {i}. {t['total_likes']} likes - {text}")
    
    print(f"\n👥 Top Threads by Participants:")
    top_by_users = sorted(threads, key=lambda t: t['unique_participants'], reverse=True)[:5]
    for i, t in enumerate(top_by_users, 1):
        text = truncate_text(t['root'].get('text', '')[:40], 40)
        print(f"   {i}. {t['unique_participants']} users - {text}")


def export_thread_markdown(thread: dict, output_file: str):
    """Export a thread to Markdown format."""
    
    def node_to_md(node: dict, depth: int = 0) -> str:
        lines = []
        indent = "  " * depth
        
        user = node.get('user_id', '?')
        date = node.get('date', '')
        text = node.get('text', '[no text]')
        likes = node.get('like_count', 0)
        reposts = node.get('repost_count', 0)
        
        if depth == 0:
            lines.append(f"## Root Post\n")
        else:
            lines.append(f"{indent}- **Reply**\n")
        
        lines.append(f"{indent}  - **User**: {user}")
        lines.append(f"{indent}  - **Date**: {date}")
        lines.append(f"{indent}  - **Likes**: {likes} | **Reposts**: {reposts}")
        lines.append(f"{indent}  - **Text**:\n")
        for text_line in text.split('\n'):
            lines.append(f"{indent}    > {text_line}")
        lines.append("")
        
        if node.get('children'):
            if depth == 0:
                lines.append(f"\n## Replies\n")
            for child in node['children']:
                lines.append(node_to_md(child, depth + 1))
        
        return "\n".join(lines)
    
    md_content = f"""# Thread {thread['thread_id']}

## Metadata
- **Thread ID**: {thread['thread_id']}
- **Source Feed**: {thread['source_feed']}
- **Depth**: {thread['depth']}
- **Total Posts**: {thread['total_posts']}
- **Unique Participants**: {thread['unique_participants']}
- **Total Likes**: {thread['total_likes']}
- **Total Reposts**: {thread['total_reposts']}
- **Root Post Date**: {thread['root_post_date']}

---

{node_to_md(thread['root'])}
"""
    
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"Exported thread to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="View and explore extracted threads")
    parser.add_argument('input_file', help='Path to threads JSONL file')
    parser.add_argument('--thread', '-t', type=str, help='View specific thread (by index or ID)')
    parser.add_argument('--full', '-f', action='store_true', help='Show full text of thread')
    parser.add_argument('--search', '-s', type=str, help='Search threads for keyword')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--export', '-e', type=str, help='Export thread to Markdown file')
    parser.add_argument('--limit', '-l', type=int, default=20, help='Limit summary output')
    parser.add_argument('--max-depth', type=int, default=10, help='Max tree depth to display')
    
    args = parser.parse_args()
    
    # Load threads
    print(f"Loading threads from {args.input_file}...")
    threads = load_threads(args.input_file)
    print(f"Loaded {len(threads)} threads")
    
    # Handle different modes
    if args.stats:
        print_statistics(threads)
    elif args.search:
        results = search_threads(threads, args.search)
        print(f"\nFound {len(results)} threads matching '{args.search}'")
        print_thread_summary(results, limit=args.limit)
    elif args.thread is not None:
        # Find thread by index or ID
        thread = None
        try:
            idx = int(args.thread)
            if idx < len(threads):
                thread = threads[idx]
            else:
                # Try as thread_id
                for t in threads:
                    if t['thread_id'] == idx:
                        thread = t
                        break
        except ValueError:
            pass
        
        if thread:
            if args.export:
                export_thread_markdown(thread, args.export)
            elif args.full:
                print_full_thread(thread)
            else:
                print_thread_detail(thread, max_depth=args.max_depth)
        else:
            print(f"Thread '{args.thread}' not found")
    else:
        print_thread_summary(threads, limit=args.limit)


if __name__ == "__main__":
    main()



