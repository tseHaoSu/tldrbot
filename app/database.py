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
    __tablename__ = "allowed_users"

    id = Column(Integer, primary_key=True)
    threads_user_id = Column(String(100), unique=True, nullable=False)
    username = Column(String(100))
    added_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class SummaryRequest(Base):
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
