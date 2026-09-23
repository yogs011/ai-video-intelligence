# AI Video Intelligence & Semantic Search Pipeline

A lightweight Python pipeline to search through video content using natural language. It extracts audio using FFmpeg, generates timestamped transcripts using OpenAI's Whisper model, and indexes the text chunks into ChromaDB for semantic search.

Instead of scrubbing through long videos manually, you can ask questions or search topics (e.g., *"where do they discuss database indexing?"*) and get the exact timestamp and context.

---

## What it does

1. **Audio Extraction**: Takes any input video (`.mp4`, `.mkv`, etc.) and extracts a 16kHz mono WAV file using FFmpeg.
2. **Transcription**: Runs Whisper (default: `base` model) to transcribe speech into timestamped chunks (`start`, `end`, `text`).
3. **Vector Indexing**: Converts transcript chunks into vector embeddings via `sentence-transformers` and stores them in ChromaDB.
4. **Semantic Search**: Searches transcript embeddings using cosine similarity to return the most relevant video timestamps.
5. **REST API**: Built with FastAPI so it can be easily connected to a frontend or integrated into other services.

---

## Project Structure

```
ai-video-intelligence/
├── app/
│   ├── __init__.py
│   ├── config.py         # Paths, model configurations, and constants
│   ├── models.py         # Pydantic schemas for requests/responses
│   ├── transcriber.py    # FFmpeg audio extraction & Whisper runner
│   ├── indexer.py        # ChromaDB setup and vector search logic
│   └── main.py           # FastAPI routes (/upload, /search, /summary)
├── tests/
│   ├── __init__.py
│   └── test_api.py       # API test cases
├── requirements.txt
├── demo.py               # Quick test script without running server
└── README.md
```

---

## Setup & Running Locally

### 1. Requirements
- Python 3.10+
- FFmpeg installed and added to PATH
  - Windows: `winget install Gyan.FFmpeg`
  - Mac: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

### 2. Install dependencies
```bash
# Clone the repository
git clone https://github.com/Yogs011/ai-video-intelligence.git
cd ai-video-intelligence

# Setup virtual environment
python -m venv venv

# Activate venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the FastAPI server
```bash
uvicorn app.main:app --reload --port 8000
```
Open your browser at `http://localhost:8000/docs` to interact with Swagger UI.

### 4. Or run the standalone demo
```bash
python demo.py
```

---

## API Endpoints

- `GET /health` - Health check endpoint.
- `POST /api/v1/videos/upload` - Upload video file, transcribe, and index into ChromaDB.
- `GET /api/v1/search?q=<query>` - Search across indexed video transcripts and return timestamp matches.
- `GET /api/v1/videos/{video_id}/summary` - Get quick summary and key takeaways of the video.

---

## Design Choices & Notes

- **Why ChromaDB?** Needed an embedded vector database that runs locally without needing an external Docker container or paid cloud service like Pinecone.
- **Why Whisper `base`?** The `base` model provides a good balance between transcription accuracy and CPU inference speed (~2-3x faster than `small`/`medium` on laptops without dedicated GPU).
- **Audio Pre-processing**: Downsampling audio to 16kHz mono WAV speeds up Whisper processing and reduces memory overhead significantly.

---

## Author
**Yogesh Birla**  
- GitHub: [Yogs011](https://github.com/Yogs011)
- LinkedIn: [Yogesh Birla](https://www.linkedin.com/in/yogesh-birla-374b59251)
