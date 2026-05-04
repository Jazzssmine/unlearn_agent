# Thread Extraction for Bluesky Social Dataset

This module extracts conversation threads from the [Bluesky Social Dataset](https://zenodo.org/records/14669616) and saves them in a structured, hierarchical format.

## Overview

The Bluesky dataset contains ~235M posts from ~4M users, but the posts are scattered across individual user files. This module:

1. Identifies root posts from a specific feed (e.g., News)
2. Finds all participants in each thread using the pre-computed `threads.txt`
3. Collects all posts belonging to each thread from user files
4. Builds hierarchical tree structures with parent-child relationships
5. Filters threads by depth and saves comprehensive JSON output
6. Processes nested threads to extract conversation chains for simulation (OP → Author 1 → Author 2 → [Optional OP Return])

## Quick Start

### Extract Threads

```bash
# Extract 500 threads with depth >= 4 from News feed
python src/thread_extraction/extract_news_threads.py \
    --output data/news_threads.jsonl \
    --num-threads 500 \
    --min-depth 4
```

### View Extracted Threads

```bash
# View summary of all threads
python src/thread_extraction/view_threads.py data/news_threads.jsonl

# View statistics
python src/thread_extraction/view_threads.py data/news_threads.jsonl --stats

# View a specific thread (by index)
python src/thread_extraction/view_threads.py data/news_threads.jsonl --thread 0

# View full text of a thread
python src/thread_extraction/view_threads.py data/news_threads.jsonl --thread 0 --full

# Search threads
python src/thread_extraction/view_threads.py data/news_threads.jsonl --search "climate"

# Export thread to Markdown
python src/thread_extraction/view_threads.py data/news_threads.jsonl --thread 0 --export thread.md
```

### Process Threads to CSV

```bash
# Convert nested thread JSON files to CSV format for simulation
# Processes all .json and .jsonl files in the input directory
python src/thread_extraction/process_threads.py \
    --input_dir ./data/ \
    --output_file ./data/threads_data.csv
```

## Output Format

### Main Output: `news_threads.jsonl`

Each line is a JSON object representing one thread:

```json
{
  "thread_id": 12345,
  "source_feed": "News",
  "depth": 5,
  "total_posts": 23,
  "unique_participants": 15,
  "participant_ids": [123, 456, 789, ...],
  "total_likes": 1234,
  "total_reposts": 56,
  "root_post_date": "2024-03-15T10:30:00",
  "root": {
    "post_id": 12345,
    "user_id": 123,
    "text": "Original post content...",
    "date": "2024-03-15T10:30:00",
    "date_raw": 202403151030,
    "langs": ["eng"],
    "like_count": 100,
    "reply_count": 22,
    "repost_count": 20,
    "instance": "bsky.social",
    "labels": null,
    "sentiment_label": null,
    "sentiment_score": null,
    "children": [
      {
        "post_id": 12346,
        "user_id": 456,
        "text": "First reply...",
        "children": [
          {
            "post_id": 12350,
            "user_id": 789,
            "text": "Nested reply...",
            "children": []
          }
        ]
      },
      {
        "post_id": 12347,
        "user_id": 111,
        "text": "Second reply...",
        "children": []
      }
    ]
  }
}
```

### Summary Output: `news_threads.summary.json`

```json
{
  "source_feed": "News",
  "extraction_date": "2024-03-15T12:00:00",
  "total_threads": 500,
  "min_depth_filter": 4,
  "depth_distribution": {
    "4": 150,
    "5": 120,
    "6": 80,
    "7": 50,
    ...
  },
  "avg_posts_per_thread": 15.5,
  "avg_participants_per_thread": 10.2,
  "total_posts_extracted": 7750,
  "total_unique_users": 3200
}
```

### Sample Output: `news_threads.sample.json`

A single thread in pretty-printed JSON for easy inspection.

### CSV Output: `threads_data.csv`

Processed conversation chains in CSV format for simulation. Each row represents one step in a conversation chain:

| Column | Description |
|--------|-------------|
| `thread_id` | Unique identifier for the conversation chain (format: `{op_post_id}_{a1_post_id}_{a2_post_id}`) |
| `sequence` | Position in the chain (1=OP, 2=Author 1, 3=Author 2, 4=OP Return if present) |
| `author_id` | Anonymized user ID of the author |
| `content` | Text content of the post |

Example:
```csv
thread_id,sequence,author_id,content
12345_12346_12350,1,39699,"Original post content..."
12345_12346_12350,2,456,"First reply..."
12345_12346_12350,3,789,"Nested reply..."
12345_12346_12350,4,39699,"OP's response back..."
```

**Chain Pattern**: The script extracts chains following the pattern:
- **Sequence 1**: Root post (OP)
- **Sequence 2**: Direct reply to OP (Author 1)
- **Sequence 3**: Reply to Author 1 (Author 2)
- **Sequence 4** (optional): OP's reply back to Author 2

Each thread can produce multiple chains if there are multiple reply branches.

## Thread Structure

### Understanding Depth

Thread depth is the maximum number of levels in the conversation tree:

```
Depth 1: Root post only (no replies)
Depth 2: Root → Reply
Depth 3: Root → Reply → Nested reply
Depth 4: Root → Reply → Nested → Nested again
...
```

Example of a depth-4 thread:
```
Post A (root)                    ← Level 1
├── Post B (reply to A)          ← Level 2
│   └── Post C (reply to B)      ← Level 3
│       └── Post D (reply to C)  ← Level 4
└── Post E (reply to A)          ← Level 2
```

### Post Fields

| Field | Description |
|-------|-------------|
| `post_id` | Unique identifier for the post |
| `user_id` | Anonymized author ID |
| `text` | Post content |
| `date` | ISO format timestamp |
| `date_raw` | Original timestamp (YYYYMMDDHHmm) |
| `langs` | Detected languages |
| `like_count` | Number of likes |
| `reply_count` | Number of direct replies |
| `repost_count` | Number of reposts |
| `instance` | Bluesky instance |
| `children` | Array of reply posts (recursive) |

## Data Sources

The extraction uses these files from the Bluesky dataset:

| File | Purpose |
|------|---------|
| `feed_posts/News.jsonl` | Root posts from News feed |
| `graphs/threads.txt` | Thread participant lists |
| `user_posts/{user_id}.jsonl` | Individual user posts |

## Algorithm

### Thread Extraction (`extract_news_threads.py`)

1. **Load root posts**: Read News.jsonl, keep posts with `reply_to=null` and `reply_count>0`
2. **Get participants**: Scan threads.txt to find user_ids for each thread
3. **Collect posts**: For each participant, scan their posts file for matching `thread_root`
4. **Build trees**: Use `reply_to` field to construct parent-child relationships
5. **Filter by depth**: Calculate max depth, keep threads >= min_depth
6. **Save output**: Write threads as JSONL with metadata

### Thread Processing (`process_threads.py`)

1. **Load thread files**: Read all `.json` and `.jsonl` files from input directory
2. **Traverse nested structure**: For each thread, traverse the `root` → `children` → `children` hierarchy
3. **Extract chains**: Identify chains of pattern: Root (OP) → Child (Author 1) → Grandchild (Author 2)
4. **Check for OP return**: Look for OP's reply to Author 2 in the grandchild's children
5. **Generate chain ID**: Create unique ID from post IDs: `{op_post_id}_{a1_post_id}_{a2_post_id}`
6. **Export to CSV**: Write flattened chain data with sequence numbers

## Performance Notes

- The script scans ~19.5M lines in threads.txt (takes ~1 minute)
- User file scanning depends on number of unique participants
- Typical extraction of 500 threads: 10-30 minutes
- Output size: ~50-200 MB depending on thread complexity

## Customization

### Extract from Different Feeds

Modify `NEWS_FEED` path in `extract_news_threads.py`:

```python
# For AcademicSky feed
NEWS_FEED = DATA_DIR / "feed_posts" / "AcademicSky.jsonl"
```

### Change Depth Requirement

```bash
python extract_news_threads.py --min-depth 6  # Deeper threads only
```

### Extract More/Fewer Threads

```bash
python extract_news_threads.py --num-threads 1000  # More threads
```

## Files

```
src/thread_extraction/
├── __init__.py
├── extract_news_threads.py  # Main extraction script (JSON → JSONL)
├── process_threads.py       # Thread processing script (JSON/JSONL → CSV)
├── view_threads.py          # Thread viewer utility
└── README.md                # This file

data/
├── news_threads.jsonl       # Extracted threads (JSONL)
├── news_threads.summary.json # Extraction statistics
├── news_threads.sample.json  # Sample thread (pretty JSON)
└── threads_data.csv         # Processed conversation chains (CSV)
```

## Workflow

The typical workflow is:

1. **Extract threads** from Bluesky dataset:
   ```bash
   python src/thread_extraction/extract_news_threads.py --output data/news_threads.jsonl --num-threads 500 --min-depth 4
   ```

2. **Process threads** to CSV format for simulation:
   ```bash
   python src/thread_extraction/process_threads.py --input_dir ./data/ --output_file ./data/threads_data.csv
   ```

3. **View/Inspect** threads as needed:
   ```bash
   python src/thread_extraction/view_threads.py data/news_threads.jsonl --stats
   ```

## Citation

If you use this data for research, please cite the original dataset:

> Andrea Failla and Giulio Rossetti. "I'm in the Bluesky Tonight: Insights from a Year's Worth of Social Data." PlosOne (2024) https://doi.org/10.1371/journal.pone.0310330



