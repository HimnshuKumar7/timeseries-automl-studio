from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from .auth import create_session_token, hash_password, verify_password, verify_session_token
from .config import CORS_ORIGINS, IS_PRODUCTION, SESSION_COOKIE, SESSION_MAX_AGE_SECONDS
from .db import close_pool, get_pool, init_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="Time-Series AutoML Studio API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # The original Next.js routes returned { error: "..." } — keep that
    # contract instead of FastAPI's default { detail: "..." } so the
    # frontend's `data.error` reads need no changes.
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"error": "Invalid request."})


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=IS_PRODUCTION,
        samesite="lax",
        path="/",
        max_age=SESSION_MAX_AGE_SECONDS,
    )


class SignupBody(BaseModel):
    name: str
    email: EmailStr
    password: str
    gender: Optional[str] = None


class LoginBody(BaseModel):
    email: EmailStr
    password: str


@app.post("/api/auth/signup")
async def signup(body: SignupBody, response: Response):
    if not body.name or not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required.")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    pool = get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", body.email)
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

        password_hash = hash_password(body.password)
        user = await conn.fetchrow(
            """
            INSERT INTO users (name, gender, email, password_hash)
            VALUES ($1, $2, $3, $4)
            RETURNING id, email
            """,
            body.name,
            body.gender,
            body.email,
            password_hash,
        )

    token = create_session_token(user["id"], user["email"])
    _set_session_cookie(response, token)
    return {"ok": True}


@app.post("/api/auth/login")
async def login(body: LoginBody, response: Response):
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    pool = get_pool()
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, password_hash FROM users WHERE email = $1", body.email
        )

    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_session_token(user["id"], user["email"])
    _set_session_cookie(response, token)
    return {"ok": True}


@app.post("/api/auth/logout")
async def logout(response: Response):
    response.set_cookie(key=SESSION_COOKIE, value="", path="/", max_age=0)
    return {"ok": True}


@app.get("/api/auth/me")
async def me(response: Response, session: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    payload = verify_session_token(session) if session else None
    if not payload:
        response.status_code = 401
        return {"user": None}

    pool = get_pool()
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, name, email FROM users WHERE id = $1", payload["userId"]
        )

    if not user:
        response.status_code = 401
        return {"user": None}

    return {"user": dict(user)}
