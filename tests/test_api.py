from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_semantic_search_empty():
    response = client.get("/api/v1/search?q=machine%20learning")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)

def test_video_summary():
    response = client.get("/api/v1/videos/demo123/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "demo123"
    assert "key_points" in data
