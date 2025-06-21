from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="Cline Context Expansion Server",
    description="Provides context expansion and vector store services for the Cline local LLM system.",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("src.integration.cline_server:app", host="0.0.0.0", port=8001, reload=False)
