"""
Quick standalone demo script to test the indexing and search logic
without having to launch the full FastAPI web server.
"""

from app.indexer import SemanticIndexer

def main():
    print("=== AI Video Intelligence Pipeline Demo ===")
    indexer = SemanticIndexer()
    
    # Sample transcript segments (simulating Whisper output)
    sample_segments = [
        {"id": 0, "start": 0.0, "end": 5.2, "text": "Welcome to the introduction on database indexing techniques."},
        {"id": 1, "start": 5.5, "end": 12.0, "text": "B-Tree and Hash indexes help optimize query execution time on large datasets."},
        {"id": 2, "start": 12.5, "end": 18.0, "text": "Next we will discuss vector databases like ChromaDB for semantic search over unstructured text."},
        {"id": 3, "start": 18.2, "end": 24.0, "text": "FastAPI provides high performance asynchronous endpoints for machine learning models."}
    ]
    
    video_id = "demo_clip_01"
    print(f"\n[1] Indexing {len(sample_segments)} transcript segments for video: {video_id}...")
    indexer.index_segments(video_id=video_id, segments=sample_segments)
    print("Indexing completed!")
    
    # Perform a natural language search query
    query = "how to optimize queries using database indexes"
    print(f"\n[2] Searching query: '{query}'...")
    results = indexer.search(query=query, top_k=2)
    
    print("\n--- Search Results ---")
    for idx, res in enumerate(results, start=1):
        print(f"Match #{idx}:")
        print(f"  Timestamp: [{res['start_time']}s - {res['end_time']}s]")
        print(f"  Snippet  : \"{res['text']}\"")
        print(f"  Score    : {res['similarity_score']}\n")

if __name__ == "__main__":
    main()
