# TLDRBot

Threads bot that summarizes posts when mentioned by allowed users.

## How It Works

1. Allowed user mentions `@tldrbot` on a post
2. Bot fetches the post content
3. Bot generates summary with OpenAI
4. Bot replies with summary

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI |
| Database | Neon Postgres |
| AI | OpenAI GPT-4o-mini |
| Deployment | Railway |
| Platform | Threads API |

## Project Structure

```
tldrbot/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app + routes
│   ├── config.py         # Settings
│   ├── database.py       # DB setup + models
│   ├── threads_api.py    # Threads API client
│   ├── openai_service.py # Summarization
│   └── bot.py            # Core logic
├── scripts/
│   ├── simulate_webhook.py
│   └── manage_allowlist.py
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- Meta Developer Account with Threads API access
- OpenAI API key
- Neon Postgres database
- Railway account (for deployment)

### Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/tldrbot.git
cd tldrbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy env file and fill in values
cp .env.example .env
```

### Environment Variables

```bash
THREADS_APP_ID=your_app_id
THREADS_APP_SECRET=your_app_secret
THREADS_ACCESS_TOKEN=your_access_token
THREADS_VERIFY_TOKEN=random_string_you_create
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+asyncpg://user:pass@ep-xxx.neon.tech/neondb?ssl=require
DEBUG=true
```

### Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

## Manage Allowlist

Only users in the allowlist can trigger the bot.

```bash
# List allowed users
python scripts/manage_allowlist.py list

# Add user
python scripts/manage_allowlist.py add <user_id> [username]

# Remove user
python scripts/manage_allowlist.py remove <user_id>
```

## Testing

```bash
# Test webhook locally
python scripts/simulate_webhook.py

# Health check
curl http://localhost:8000/
```

## Deployment

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Set environment variables
railway variables set THREADS_APP_ID=xxx
railway variables set THREADS_APP_SECRET=xxx
railway variables set THREADS_ACCESS_TOKEN=xxx
railway variables set THREADS_VERIFY_TOKEN=xxx
railway variables set OPENAI_API_KEY=xxx
railway variables set DATABASE_URL=xxx
railway variables set DEBUG=false

# Get public URL
railway domain
```

### Configure Meta Webhooks

1. Go to Meta Developer Console
2. Your App > Threads > Webhooks
3. Add URL: `https://your-railway-url/webhook`
4. Verify token: your THREADS_VERIFY_TOKEN
5. Subscribe to "mentions"

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| GET | `/webhook` | Meta verification |
| POST | `/webhook` | Receive mentions |

## Database Schema

### allowed_users

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| threads_user_id | VARCHAR(100) | Threads user ID (unique) |
| username | VARCHAR(100) | Display name |
| added_at | TIMESTAMP | When added |

### summary_requests

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| post_id | VARCHAR(100) | Post that was summarized |
| requesting_user_id | VARCHAR(100) | Who requested |
| requesting_username | VARCHAR(100) | Username |
| summary | TEXT | Generated summary |
| reply_post_id | VARCHAR(100) | Reply post ID |
| status | VARCHAR(20) | pending/posted/failed |
| created_at | TIMESTAMP | When requested |
| error_message | TEXT | Error if failed |
