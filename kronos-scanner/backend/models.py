# models.py
"""SQLAlchemy ORM models and DB session helpers for the Kronos scanner."""

import os
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class TickerMetadata(Base):
    __tablename__ = "ticker_metadata"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, index=True)
    sector = Column(String(50))
    market_cap = Column(Float)  # millions
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyMarketData(Base):
    __tablename__ = "daily_market_data"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    rsi_14 = Column(Float)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uix_ticker_date"),
        Index("ix_daily_ticker_date", "ticker", "date"),
    )


class QuantSignalOutputs(Base):
    __tablename__ = "quant_signals"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    date = Column(DateTime, index=True)
    ou_score = Column(Float)  # 0-100
    ou_halflife = Column(Float)  # days
    hmm_regime = Column(String(20))  # Bull, Bear, HighVol, LowVol
    hmm_confidence = Column(Float)  # 0-1
    kalman_alpha = Column(Float)
    evt_var_99 = Column(Float)  # VaR at 99%
    evt_expected_shortfall = Column(Float)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uix_signal_ticker_date"),
    )


def get_database_url() -> str:
    """Resolve the database URL from the environment with a local default."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://kronos:kronos@localhost:5432/kronos",
    )


def make_engine(echo: bool = False):
    return create_engine(get_database_url(), echo=echo, future=True)


def make_session_factory(engine=None):
    engine = engine or make_engine()
    return sessionmaker(bind=engine, autoflush=False, future=True)


def init_db(engine=None):
    """Create all tables. Safe to call repeatedly."""
    engine = engine or make_engine()
    Base.metadata.create_all(engine)
    return engine
