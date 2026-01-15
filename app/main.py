"""
FastAPI application entry point.

FastAPI is similar to Express.js:
- app = FastAPI()  ≈  const app = express()
- @app.get("/")    ≈  app.get("/", handler)
- async def        ≈  async (req, res) => {}
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database import engine, Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events - runs on startup and shutdown.

    Similar to:
    - Next.js: instrumentation.ts
    - Express: app.listen() callback

    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    # STARTUP
    print("Starting up...")

    # Create database tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables ready")

    yield  # App runs here

    # SHUTDOWN
    print("Shutting down...")
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="TLDRBot",
    description="Summarizes Threads posts",
    lifespan=lifespan
)


@app.get("/")
async def health():
    """
    Health check endpoint.

    Used by:
    - Railway/deployment platforms to check if app is alive
    - You, to verify the server is running
    """
    return {"status": "ok", "service": "tldrbot"}
