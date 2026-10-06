"""Alert and report tables (report section 10.3), SQLAlchemy Core on Postgres or SQLite.

`alert_truth` holds the dataset labels for evaluation only. No copilot tool can read it,
so ground truth cannot leak into an investigation.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    make_url,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

JSONType = JSON().with_variant(JSONB(), "postgresql")

metadata = MetaData()

alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("ts", DateTime, nullable=False),
    Column("source", String(16), nullable=False),  # "network" or "email"
    Column("src_ip", String(64)),
    Column("dst_ip", String(64)),
    Column("src_port", Integer),
    Column("dst_port", Integer),
    Column("protocol", Integer),
    Column("sender_domain", String(255)),
    Column("predicted_family", String(32), nullable=False),
    Column("detector", String(16)),  # supervised / anomaly / text / url
    Column("confidence", Float),
    Column("attack_score", Float),
    Column("anomaly_score", Float),
    Column("severity", String(16)),
    Column("reasons", JSONType),  # SHAP / anomaly / sentence reasons from the detector
    Column("evidence", JSONType),  # untrusted text (email subject/body, URLs, log lines)
    Column("features", JSONType),  # a few raw flow statistics (behaviour descriptions)
    Column("model_version", String(255)),
    Column("status", String(16), nullable=False, server_default="new"),
    Index("alerts_src_ts", "src_ip", "ts"),
    Index("alerts_family", "predicted_family"),
)

alert_truth = Table(
    "alert_truth",
    metadata,
    Column("alert_id", Integer, ForeignKey("alerts.id"), primary_key=True),
    Column("label", String(64), nullable=False),  # dataset sub-label, e.g. "DoS Hulk"
    Column("family", String(32), nullable=False),
)

# A random sample of scored flows (report section 10.3): the live feature window for drift
# detection and, with the label that arrives later, the data for retraining.
flows = Table(
    "flows",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("ts", DateTime, nullable=False),
    Column("src_ip", String(64)),
    Column("dst_ip", String(64)),
    Column("dst_port", Integer),
    Column("features", JSONType, nullable=False),  # the model inputs as scored
    Column("predicted_family", String(32)),
    Column("label", String(64)),  # ground truth (replay: dataset label; live: analyst feedback)
    Column("family", String(32)),
    Column("alert_id", Integer, ForeignKey("alerts.id")),
    Column("model_version", String(255)),
    Column("scored_at", DateTime, nullable=False),
    Index("flows_scored_at", "scored_at"),
)

# Module 6 MLOps: every drift check, and every retraining it triggered.
drift_checks = Table(
    "drift_checks",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("ts", DateTime, nullable=False),
    Column("window", String(16), nullable=False),  # "all" or "benign" (flows the model passed)
    Column("n", Integer, nullable=False),
    Column("share", Float),  # share of features drifted
    Column("drifted", JSONType),  # feature -> PSI, for drifted features
    Column("triggered", Integer, nullable=False, server_default="0"),
)

retrain_runs = Table(
    "retrain_runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("started", DateTime, nullable=False),
    Column("finished", DateTime),
    Column("reason", Text),
    Column("n_extra", Integer),  # labelled live flows added to training
    Column("status", String(16), nullable=False),  # running / promoted / staged / failed
    Column("version", String(32)),  # registry version created
    Column("metrics", JSONType),  # candidate vs production on the fixed test split
    Column("error", Text),
)

reports = Table(
    "reports",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("alert_id", Integer, ForeignKey("alerts.id"), nullable=False),
    Column("created", DateTime, nullable=False),
    Column("status", String(16), nullable=False),  # done / failed
    Column("provider", String(64)),
    Column("report", JSONType),
    Column("trace", JSONType),  # per-node timings, tokens, guard findings, retries
    Column("error", Text),
)

# Tables the copilot's read-only role may see.
COPILOT_TABLES = ("alerts", "reports")


def make_engine(url: str) -> Engine:
    engine = create_engine(url, future=True, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(engine: Engine, ro_password: str | None = None) -> None:
    """Create the tables; on Postgres also a SELECT-only role for the copilot."""
    metadata.create_all(engine)
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        # Columns added after the first release (create_all never alters a table).
        conn.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS features JSONB"))
    if not ro_password:
        return
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_ro'")).first()
        if not exists:
            # DDL cannot take bound parameters; the password comes from our own settings
            # and is quoted as a SQL string literal.
            literal = "'" + ro_password.replace("'", "''") + "'"
            conn.execute(text(f"CREATE ROLE sentinel_ro LOGIN PASSWORD {literal}"))
        conn.execute(text("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM sentinel_ro"))
        for t in COPILOT_TABLES:
            conn.execute(text(f"GRANT SELECT ON {t} TO sentinel_ro"))


def with_database(url: str, database: str | None) -> str:
    """The same server URL pointing at another database (e.g. the dev alert set)."""
    if not database:
        return url
    return make_url(url).set(database=database).render_as_string(hide_password=False)


def ensure_database(url: str) -> None:
    """Create the URL's Postgres database if it does not exist (SQLite creates files itself)."""
    u = make_url(url)
    if not u.drivername.startswith("postgresql") or not u.database:
        return
    admin = create_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        found = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": u.database}
        ).first()
        if not found:
            name = conn.dialect.identifier_preparer.quote(u.database)
            conn.execute(text(f"CREATE DATABASE {name}"))
    admin.dispose()
