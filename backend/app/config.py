import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgres://dummy:dummy@dummy/dummy")
JWT_SECRET = os.getenv("JWT_SECRET", "dummy_secret_for_build_only")

# Comma-separated list of origins allowed to send credentialed requests,
# e.g. "http://localhost:5173,https://app.example.com"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

SESSION_COOKIE = "session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days, matches the original '7d' JWT
