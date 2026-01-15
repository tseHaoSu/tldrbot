# @SummarizerBot - Threads AI Summarizer

## Project Plan (7-Day)

---

## Core Feature

**Only this, nothing else:**
1. Allowed user mentions `@summarizerbot` on a post
2. Bot fetches the post content
3. Bot generates summary with OpenAI
4. Bot replies with summary

**Access Control:** Only manually added users in allowlist can trigger the bot.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI |
| Database | Neon Postgres |
| AI | OpenAI GPT-4o-mini |
| Deployment | Railway |
| Platform | Threads API |

---

## Prerequisites (Start BEFORE Day 1)

### 1. Meta Developer Account (1-3 days)
- https://developers.meta.com
- Create App → "Other" → "Access the Threads API"
- Submit for business verification

### 2. Bot Threads Account
- Create Instagram account → Sign up for Threads
- Username: `summarizerbot`

### 3. OpenAI API Key
- https://platform.openai.com
- Create API key, add $5 credit

### 4. Neon Postgres
- https://neon.tech
- Create project, copy connection string

### 5. Railway Account
- https://railway.app

### 6. Dev Environment
- Python 3.11+
- ngrok for local testing

---

## Architecture

```
Allowed user mentions @bot
         ↓
Threads webhook → FastAPI
         ↓
Check allowlist (DB)
         ↓
If allowed: Fetch post → OpenAI → Reply
         ↓
Log to database
```

---

## Database Schema

```sql
-- Allowlist: only these users can trigger the bot
CREATE TABLE allowed_users (
    id SERIAL PRIMARY KEY,
    threads_user_id VARCHAR(100) NOT NULL UNIQUE,
    username VARCHAR(100),
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Request log
CREATE TABLE summary_requests (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(100) NOT NULL,
    requesting_user_id VARCHAR(100) NOT NULL,
    requesting_username VARCHAR(100),
    summary TEXT,
    reply_post_id VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error_message TEXT
);
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| GET | `/webhook` | Meta verification |
| POST | `/webhook` | Receive mentions |

---

## Project Structure

```
threads-summarizer-bot/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app + routes
│   ├── config.py         # Settings
│   ├── database.py       # DB setup + models
│   ├── threads_api.py    # Threads API client
│   ├── openai_service.py # Summarization
│   └── bot.py            # Core logic (allowlist check + process)
├── scripts/
│   └── simulate_webhook.py
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Day 1: FastAPI + Database Setup

### Tasks

#### 1.1: Create Project
```bash
mkdir threads-summarizer-bot
cd threads-summarizer-bot
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

#### 1.2: Install Dependencies
```bash
pip install fastapi uvicorn sqlalchemy asyncpg python-dotenv httpx openai pydantic-settings
pip freeze > requirements.txt
```

#### 1.3: Create Config

**File: `app/config.py`**
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Threads API
    threads_app_id: str
    threads_app_secret: str
    threads_access_token: str
    threads_verify_token: str

    # OpenAI
    openai_api_key: str

    # Database
    database_url: str

    # Debug
    debug: bool = False

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**File: `.env.example`**
```bash
THREADS_APP_ID=your_app_id
THREADS_APP_SECRET=your_app_secret
THREADS_ACCESS_TOKEN=your_access_token
THREADS_VERIFY_TOKEN=my_random_verify_token_12345
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+asyncpg://user:pass@ep-xxx.neon.tech/neondb?ssl=require
DEBUG=true
```

#### 1.4: Create Database

**File: `app/database.py`**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import TIMESTAMP
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()


class AllowedUser(Base):
    """Users allowed to trigger the bot"""
    __tablename__ = "allowed_users"

    id = Column(Integer, primary_key=True)
    threads_user_id = Column(String(100), unique=True, nullable=False)
    username = Column(String(100))
    added_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class SummaryRequest(Base):
    """Log of summary requests"""
    __tablename__ = "summary_requests"

    id = Column(Integer, primary_key=True)
    post_id = Column(String(100), nullable=False)
    requesting_user_id = Column(String(100), nullable=False)
    requesting_username = Column(String(100))
    summary = Column(Text)
    reply_post_id = Column(String(100))
    status = Column(String(20), default="pending")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    error_message = Column(Text)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

#### 1.5: Create Main App

**File: `app/main.py`**
```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database ready")
    yield
    await engine.dispose()

app = FastAPI(title="SummarizerBot", lifespan=lifespan)

@app.get("/")
async def health():
    return {"status": "ok"}
```

### Day 1 Testing

```bash
# Start server
uvicorn app.main:app --reload --port 8000

# Test health
curl http://localhost:8000/
# Expected: {"status":"ok"}

# Check Neon dashboard for tables
# Should see: allowed_users, summary_requests
```

---

## Day 2: Webhook System

### Tasks

#### 2.1: Add Webhook Routes

**Update `app/main.py`:**
```python
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
import hmac
import hashlib
import json
from app.database import engine, Base
from app.config import get_settings
from app.bot import process_mention

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database ready")
    yield
    await engine.dispose()

app = FastAPI(title="SummarizerBot", lifespan=lifespan)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = None,
    hub_verify_token: str = None,
    hub_challenge: str = None
):
    """Meta webhook verification"""
    if hub_mode == "subscribe" and hub_verify_token == settings.threads_verify_token:
        print("✅ Webhook verified")
        return PlainTextResponse(content=hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify Meta's webhook signature"""
    if not signature:
        return False
    expected = "sha256=" + hmac.new(
        key=settings.threads_app_secret.encode(),
        msg=payload,
        digestmod=hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive mention events from Threads"""
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body)
    print(f"📨 Webhook received")

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") == "mentions":
                background_tasks.add_task(
                    process_mention,
                    change.get("value", {})
                )

    return {"status": "received"}
```

#### 2.2: Create Bot Logic (Placeholder)

**File: `app/bot.py`**
```python
"""Core bot logic"""
from datetime import datetime

async def process_mention(mention_data: dict):
    """Process a mention - placeholder for now"""
    post_id = mention_data.get("thread_id")
    user_id = mention_data.get("mentioning_user_id")
    username = mention_data.get("mentioning_username", "unknown")

    print(f"\n{'='*50}")
    print(f"🔔 Mention received at {datetime.now()}")
    print(f"   Post: {post_id}")
    print(f"   User: @{username} ({user_id})")
    print(f"{'='*50}\n")

    # TODO: Check allowlist
    # TODO: Fetch post
    # TODO: Generate summary
    # TODO: Reply
```

#### 2.3: Create Mock Webhook Script

**File: `scripts/simulate_webhook.py`**
```python
"""Simulate Threads webhook for testing"""
import httpx
import hmac
import hashlib
import json

WEBHOOK_URL = "http://localhost:8000/webhook"
APP_SECRET = "your_app_secret_here"  # Match your .env

def send_mock_webhook(post_id: str, user_id: str, username: str):
    payload = {
        "object": "threads",
        "entry": [{
            "id": "bot_id",
            "time": 1704067200,
            "changes": [{
                "field": "mentions",
                "value": {
                    "thread_id": post_id,
                    "mention_id": f"mention_{post_id}",
                    "mentioning_user_id": user_id,
                    "mentioning_username": username
                }
            }]
        }]
    }

    payload_bytes = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(
        APP_SECRET.encode(),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

    response = httpx.post(
        WEBHOOK_URL,
        content=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": signature
        }
    )
    print(f"Response: {response.status_code} - {response.text}")

if __name__ == "__main__":
    send_mock_webhook("post_123", "user_456", "testuser")
```

### Day 2 Testing

```bash
# Terminal 1: Run server
uvicorn app.main:app --reload

# Terminal 2: Test verification
curl "http://localhost:8000/webhook?hub_mode=subscribe&hub_verify_token=YOUR_TOKEN&hub_challenge=test123"
# Expected: test123

# Terminal 3: Test mock webhook (update APP_SECRET first)
python scripts/simulate_webhook.py
# Expected: Server logs show "🔔 Mention received"
```

---

## Day 3: Threads API Client

### Tasks

#### 3.1: Create Threads API Client

**File: `app/threads_api.py`**
```python
"""Threads API client"""
import httpx
from app.config import get_settings

settings = get_settings()


class ThreadsAPI:
    """Client for Threads API"""

    BASE_URL = "https://graph.threads.net/v1.0"

    def __init__(self):
        self.token = settings.threads_access_token
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_post(self, post_id: str) -> dict:
        """Fetch a post's content"""
        url = f"{self.BASE_URL}/{post_id}"
        params = {
            "fields": "id,text,username,timestamp,permalink",
            "access_token": self.token
        }

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def reply(self, reply_to_id: str, text: str) -> dict:
        """Post a reply"""
        # Step 1: Create container
        url = f"{self.BASE_URL}/me/threads"
        params = {
            "media_type": "TEXT",
            "text": text,
            "reply_to_id": reply_to_id,
            "access_token": self.token
        }

        response = await self.client.post(url, params=params)
        response.raise_for_status()
        container_id = response.json().get("id")

        # Step 2: Publish
        url = f"{self.BASE_URL}/me/threads_publish"
        params = {
            "creation_id": container_id,
            "access_token": self.token
        }

        response = await self.client.post(url, params=params)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()


# Mock client for testing
class MockThreadsAPI:
    """Mock client for testing without real API"""

    async def get_post(self, post_id: str) -> dict:
        return {
            "id": post_id,
            "text": "This is a long post about AI and technology. It discusses various aspects of machine learning, neural networks, and their applications in modern software development. The author shares insights about best practices and future trends.",
            "username": "testuser",
            "timestamp": "2024-01-01T00:00:00+0000",
            "permalink": f"https://threads.net/post/{post_id}"
        }

    async def reply(self, reply_to_id: str, text: str) -> dict:
        print(f"[MOCK] Reply to {reply_to_id}: {text[:100]}...")
        return {"id": f"reply_{reply_to_id}"}

    async def close(self):
        pass


def get_threads_client():
    """Get Threads client (mock in debug mode)"""
    if settings.debug:
        return MockThreadsAPI()
    return ThreadsAPI()
```

### Day 3 Testing

```bash
python -c "
import asyncio
from app.threads_api import get_threads_client

async def test():
    client = get_threads_client()
    post = await client.get_post('test_123')
    print(f'Post text: {post[\"text\"][:50]}...')

    reply = await client.reply('test_123', 'Hello!')
    print(f'Reply ID: {reply[\"id\"]}')

asyncio.run(test())
"
# Expected: Shows mock post content and reply
```

---

## Day 4: OpenAI Integration

### Tasks

#### 4.1: Create OpenAI Service

**File: `app/openai_service.py`**
```python
"""OpenAI summarization service"""
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You summarize social media posts concisely.

Rules:
- 2-3 sentences max
- Capture the main point
- Neutral tone
- No emojis

Format: Just the summary, nothing else."""


class Summarizer:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def summarize(self, text: str) -> str:
        """Generate summary for post text"""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this post:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content


class MockSummarizer:
    """Mock summarizer for testing"""

    async def summarize(self, text: str) -> str:
        return f"This post discusses a topic in {len(text)} characters. The author shares their perspective and invites discussion."


def get_summarizer():
    """Get summarizer (mock in debug mode)"""
    if settings.debug:
        return MockSummarizer()
    return Summarizer()
```

#### 4.2: Complete Bot Logic

**Update `app/bot.py`:**
```python
"""Core bot logic"""
from datetime import datetime
from sqlalchemy import select
from app.database import AsyncSessionLocal, AllowedUser, SummaryRequest
from app.threads_api import get_threads_client
from app.openai_service import get_summarizer


async def is_user_allowed(user_id: str) -> bool:
    """Check if user is in allowlist"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(AllowedUser).where(AllowedUser.threads_user_id == user_id)
        )
        return result.scalar_one_or_none() is not None


async def log_request(post_id: str, user_id: str, username: str,
                      summary: str = None, reply_id: str = None,
                      status: str = "pending", error: str = None):
    """Log request to database"""
    async with AsyncSessionLocal() as db:
        request = SummaryRequest(
            post_id=post_id,
            requesting_user_id=user_id,
            requesting_username=username,
            summary=summary,
            reply_post_id=reply_id,
            status=status,
            error_message=error
        )
        db.add(request)
        await db.commit()


async def process_mention(mention_data: dict):
    """Process a mention webhook event"""
    post_id = mention_data.get("thread_id")
    mention_id = mention_data.get("mention_id")
    user_id = mention_data.get("mentioning_user_id")
    username = mention_data.get("mentioning_username", "unknown")

    print(f"\n{'='*50}")
    print(f"🔔 Processing mention from @{username}")
    print(f"{'='*50}")

    # Step 1: Check allowlist
    if not await is_user_allowed(user_id):
        print(f"❌ User {user_id} not in allowlist, ignoring")
        return

    print(f"✅ User is allowed")

    try:
        # Step 2: Fetch post
        print(f"📥 Fetching post {post_id}...")
        client = get_threads_client()
        post = await client.get_post(post_id)
        post_text = post.get("text", "")
        print(f"✅ Got post ({len(post_text)} chars)")

        # Step 3: Generate summary
        print(f"🤖 Generating summary...")
        summarizer = get_summarizer()
        summary = await summarizer.summarize(post_text)
        print(f"✅ Summary: {summary[:50]}...")

        # Step 4: Reply
        print(f"📤 Posting reply...")
        reply_text = f"📋 Summary:\n\n{summary}"

        # Truncate if needed (500 char limit)
        if len(reply_text) > 500:
            reply_text = reply_text[:497] + "..."

        reply = await client.reply(mention_id, reply_text)
        reply_id = reply.get("id")
        print(f"✅ Reply posted: {reply_id}")

        # Log success
        await log_request(post_id, user_id, username, summary, reply_id, "posted")

        await client.close()
        print(f"✅ Done!")

    except Exception as e:
        print(f"❌ Error: {e}")
        await log_request(post_id, user_id, username, error=str(e), status="failed")
```

### Day 4 Testing

```bash
# First, add a test user to allowlist manually
python -c "
import asyncio
from app.database import AsyncSessionLocal, AllowedUser

async def add_user():
    async with AsyncSessionLocal() as db:
        user = AllowedUser(threads_user_id='user_456', username='testuser')
        db.add(user)
        await db.commit()
        print('✅ User added to allowlist')

asyncio.run(add_user())
"

# Then test full flow
python scripts/simulate_webhook.py
# Expected:
# ✅ User is allowed
# ✅ Got post
# ✅ Summary: ...
# ✅ Reply posted
# ✅ Done!

# Test with non-allowed user
python -c "
import asyncio
from app.bot import process_mention

asyncio.run(process_mention({
    'thread_id': 'post_999',
    'mention_id': 'mention_999',
    'mentioning_user_id': 'unknown_user',
    'mentioning_username': 'stranger'
}))
"
# Expected: ❌ User not in allowlist, ignoring
```

---

## Day 5: Allowlist Management

### Tasks

#### 5.1: Add Helper Script for Allowlist

**File: `scripts/manage_allowlist.py`**
```python
"""Manage allowed users"""
import asyncio
import sys
from app.database import AsyncSessionLocal, AllowedUser
from sqlalchemy import select, delete

async def add_user(user_id: str, username: str = None):
    async with AsyncSessionLocal() as db:
        existing = await db.execute(
            select(AllowedUser).where(AllowedUser.threads_user_id == user_id)
        )
        if existing.scalar_one_or_none():
            print(f"User {user_id} already exists")
            return

        user = AllowedUser(threads_user_id=user_id, username=username)
        db.add(user)
        await db.commit()
        print(f"✅ Added: {user_id} ({username})")

async def remove_user(user_id: str):
    async with AsyncSessionLocal() as db:
        await db.execute(
            delete(AllowedUser).where(AllowedUser.threads_user_id == user_id)
        )
        await db.commit()
        print(f"✅ Removed: {user_id}")

async def list_users():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(AllowedUser))
        users = result.scalars().all()

        if not users:
            print("No allowed users")
            return

        print(f"\nAllowed Users ({len(users)}):")
        print("-" * 40)
        for u in users:
            print(f"  {u.threads_user_id} - @{u.username or 'unknown'}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_allowlist.py list")
        print("  python manage_allowlist.py add <user_id> [username]")
        print("  python manage_allowlist.py remove <user_id>")
        return

    cmd = sys.argv[1]

    if cmd == "list":
        asyncio.run(list_users())
    elif cmd == "add" and len(sys.argv) >= 3:
        user_id = sys.argv[2]
        username = sys.argv[3] if len(sys.argv) > 3 else None
        asyncio.run(add_user(user_id, username))
    elif cmd == "remove" and len(sys.argv) >= 3:
        asyncio.run(remove_user(sys.argv[2]))
    else:
        print("Invalid command")

if __name__ == "__main__":
    main()
```

### Day 5 Testing

```bash
# List users
python scripts/manage_allowlist.py list

# Add user
python scripts/manage_allowlist.py add "12345678" "myusername"

# Remove user
python scripts/manage_allowlist.py remove "12345678"

# Full integration test
python scripts/manage_allowlist.py add "user_456" "testuser"
python scripts/simulate_webhook.py
# Should process successfully

python scripts/manage_allowlist.py remove "user_456"
python scripts/simulate_webhook.py
# Should be ignored (not in allowlist)
```

---

## Day 6: Error Handling + Cleanup

### Tasks

#### 6.1: Add Basic Error Handling

**Update `app/bot.py`** - add better error messages:
```python
async def process_mention(mention_data: dict):
    """Process a mention webhook event"""
    post_id = mention_data.get("thread_id")
    mention_id = mention_data.get("mention_id")
    user_id = mention_data.get("mentioning_user_id")
    username = mention_data.get("mentioning_username", "unknown")

    print(f"\n{'='*50}")
    print(f"🔔 Processing mention from @{username}")
    print(f"{'='*50}")

    # Validate required fields
    if not post_id or not mention_id or not user_id:
        print(f"❌ Missing required fields in webhook")
        return

    # Check allowlist
    if not await is_user_allowed(user_id):
        print(f"❌ User {user_id} not in allowlist")
        return

    print(f"✅ User is allowed")
    client = None

    try:
        # Fetch post
        print(f"📥 Fetching post...")
        client = get_threads_client()
        post = await client.get_post(post_id)
        post_text = post.get("text", "")

        if not post_text:
            print(f"❌ Post has no text content")
            await log_request(post_id, user_id, username, error="Empty post", status="failed")
            return

        print(f"✅ Got post ({len(post_text)} chars)")

        # Generate summary
        print(f"🤖 Generating summary...")
        summarizer = get_summarizer()
        summary = await summarizer.summarize(post_text)

        if not summary:
            print(f"❌ Failed to generate summary")
            await log_request(post_id, user_id, username, error="Summary failed", status="failed")
            return

        print(f"✅ Summary generated")

        # Reply
        print(f"📤 Posting reply...")
        reply_text = f"📋 Summary:\n\n{summary}"
        if len(reply_text) > 500:
            reply_text = reply_text[:497] + "..."

        reply = await client.reply(mention_id, reply_text)
        reply_id = reply.get("id")
        print(f"✅ Reply posted: {reply_id}")

        await log_request(post_id, user_id, username, summary, reply_id, "posted")
        print(f"✅ Done!")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        await log_request(post_id, user_id, username, error=error_msg, status="failed")

    finally:
        if client:
            await client.close()
```

#### 6.2: Create .gitignore

**File: `.gitignore`**
```
venv/
__pycache__/
*.pyc
.env
*.db
.DS_Store
```

#### 6.3: Create README

**File: `README.md`**
```markdown
# @SummarizerBot

A Threads bot that summarizes posts when mentioned by allowed users.

## Setup

1. Copy `.env.example` to `.env` and fill in values
2. Install: `pip install -r requirements.txt`
3. Run: `uvicorn app.main:app --reload`

## Manage Allowlist

```bash
python scripts/manage_allowlist.py list
python scripts/manage_allowlist.py add <user_id> [username]
python scripts/manage_allowlist.py remove <user_id>
```

## Deploy

Deploy to Railway with the included Dockerfile.
```

### Day 6 Testing

```bash
# Test with missing fields
python -c "
import asyncio
from app.bot import process_mention
asyncio.run(process_mention({}))
"
# Expected: ❌ Missing required fields

# Test full flow still works
python scripts/manage_allowlist.py add "user_456" "testuser"
python scripts/simulate_webhook.py
# Expected: Full success flow
```

---

## Day 7: Deployment

### Tasks

#### 7.1: Create Dockerfile

**File: `Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 7.2: Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and init
railway login
railway init

# Set environment variables
railway variables set THREADS_APP_ID=xxx
railway variables set THREADS_APP_SECRET=xxx
railway variables set THREADS_ACCESS_TOKEN=xxx
railway variables set THREADS_VERIFY_TOKEN=xxx
railway variables set OPENAI_API_KEY=xxx
railway variables set DATABASE_URL=xxx
railway variables set DEBUG=false

# Deploy
railway up

# Get URL
railway domain
```

#### 7.3: Configure Meta Webhooks

1. Go to Meta Developer Console
2. Your App → Threads → Webhooks
3. Add URL: `https://your-railway-url/webhook`
4. Verify token: your THREADS_VERIFY_TOKEN
5. Subscribe to "mentions"

#### 7.4: Add Allowed Users

```bash
# Connect to Railway and run
railway run python scripts/manage_allowlist.py add "YOUR_USER_ID" "your_username"
```

### Day 7 Testing

```bash
# Health check
curl https://your-railway-url/

# On Threads:
# 1. From an allowed account
# 2. Reply to any post with: @summarizerbot
# 3. Wait for summary reply

# Check Railway logs
railway logs
```

---

## Summary

### What We Built

- FastAPI webhook receiver
- Allowlist-based access control
- Threads API integration (fetch post, reply)
- OpenAI summarization
- Minimal database logging

### Files Created

```
threads-summarizer-bot/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── threads_api.py
│   ├── openai_service.py
│   └── bot.py
├── scripts/
│   ├── simulate_webhook.py
│   └── manage_allowlist.py
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

### Commands Reference

```bash
# Run locally
uvicorn app.main:app --reload

# Manage allowlist
python scripts/manage_allowlist.py list
python scripts/manage_allowlist.py add <user_id> [username]
python scripts/manage_allowlist.py remove <user_id>

# Test webhook
python scripts/simulate_webhook.py

# Deploy
railway up
```
