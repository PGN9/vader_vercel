from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.responses import JSONResponse
import os
import time
import psutil
import traceback
import platform
import json

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

def get_size_in_kb(data):
    return len(data.encode('utf-8')) / 1024  # size in KB

@app.get("/")
def root():
    return {"message": "vader backend is running."}

@app.post("/predict")
def predict_sentiment(request: CommentsRequest):
    try:
        # Time start
        time_start = time.perf_counter()

        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        input_json = json.dumps(request.model_dump())
        total_data_size_kb = get_size_in_kb(input_json)

        BATCH_SIZE = 100
        comments = request.comments
        results = []

        for i in range(0, len(comments), BATCH_SIZE):
            batch = comments[i:i + BATCH_SIZE]
            for comment in batch:
                scores = analyzer.polarity_scores(comment.body)
                sentiment = (
                    "positive" if scores["compound"] > 0.05 else
                    "negative" if scores["compound"] < -0.05 else
                    "neutral"
                )
                results.append({
                    "id": comment.id,
                    "body": comment.body,
                    "sentiment": sentiment,
                    "sentiment_score": scores["compound"]
                })

        # Final memory read
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = current_memory_mb

        if platform.system() == "Linux":
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        elif platform.system() == "Darwin":
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

        elapsed_time = time.perf_counter() - time_start

        return_data = {
            "model_used": "vader",
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2),
            "processing_time_seconds": round(elapsed_time, 3),
            "total_data_size_kb": round(total_data_size_kb, 2),
            "total_return_size_kb": round(get_size_in_kb(json.dumps(results)), 2)
        }

        return return_data

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
