
from fastapi import APIRouter, UploadFile, HTTPException,Request

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/index")
async def index_document(
     request: Request,  # access app.state
    file: UploadFile | None = None,
    url: str | None = None
):
    pipeline = request.app.state.rag_pipeline 
    try:
        ids = await pipeline.ingest(file=file, url=url)
        return {"status": "indexed", "chunks":ids}
    except Exception as e:
        raise HTTPException(400, str(e))
