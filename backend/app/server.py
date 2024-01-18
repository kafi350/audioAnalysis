import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.audio import handler as audio_handler
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import DB_URL

log = logging.getLogger(__name__)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def some_middleware(request: Request, call_next):
    # Define the engine (replace 'sqlite:///foo.db' with your database connection string)
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False}, echo=True)
    # Define the session class
    Session = sessionmaker(bind=engine)
    s = Session()
    response = await call_next(request)
    s.close()
    return response


api_router = APIRouter(
    prefix="/api",
    tags=["api"],
)

api_router.include_router(audio_handler.router)

app.include_router(api_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    log.exception(exc)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )