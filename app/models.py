from pydantic import BaseModel, Field
from typing import List, Optional

class TranscriptSegment(BaseModel):
    id: int
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Transcribed text snippet")

class ProcessVideoResponse(BaseModel):
    video_id: str
    filename: str
    total_duration_seconds: Optional[float] = None
    language_detected: str
    segments_indexed: int
    full_transcript: str
    message: str

class SearchResultItem(BaseModel):
    video_id: str
    text: str
    start_time: float
    end_time: float
    similarity_score: float

class SearchResponse(BaseModel):
    query: str
    total_matches: int
    results: List[SearchResultItem]

class VideoSummaryResponse(BaseModel):
    video_id: str
    key_points: List[str]
    summary_text: str
