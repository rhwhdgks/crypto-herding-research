from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict

import pandas as pd

from utils import timeframe_to_minutes


LOGGER = logging.getLogger(__name__)


def export_research_outputs_to_database(config: dict, frame_map: Dict[str, pd.DataFrame | pd.Series]) -> None:
    database_cfg = config.get("database", {})
    if not database_cfg.get("enabled", False):
        LOGGER.info("Database export is disabled in config.")
        return

    engine = _build_sqlalchemy_engine(database_cfg)
    schema_prefix = str(database_cfg.get("table_prefix", "crypto_herding")).strip("_")
    if_exists = str(database_cfg.get("if_exists", "replace")).lower()
    chunk_size = int(database_cfg.get("chunk_size", 5000))
    insert_method = str(database_cfg.get("insert_method", "default")).lower()
    to_sql_method = "multi" if insert_method == "multi" else None

    try:
        for logical_name, frame in frame_map.items():
            if frame is None:
                continue

            export_frame = _prepare_frame_for_sql(frame)
            if export_frame.empty and not database_cfg.get("export_empty_tables", False):
                LOGGER.info("Skipping empty frame for database export: %s", logical_name)
                continue

            table_name = _build_table_name(schema_prefix, logical_name)
            LOGGER.info("Exporting %s rows to database table %s.", len(export_frame), table_name)
            export_frame.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=False,
                chunksize=chunk_size,
                method=to_sql_method,
            )
    finally:
        engine.dispose()


def export_raw_ohlcv_to_database(
    config: dict,
    asset_frames: Dict[str, pd.DataFrame],
    timeframe: str,
) -> None:
    database_cfg = config.get("database", {})
    if not database_cfg.get("enabled", False):
        LOGGER.info("Database export is disabled in config.")
        return

    if not database_cfg.get("export_raw_ohlcv", True):
        LOGGER.info("Raw OHLCV database export is disabled in config.")
        return

    connection = _build_pymysql_connection(database_cfg)
    table_name = _build_table_name(
        str(database_cfg.get("table_prefix", "crypto_herding")).strip("_"),
        f"raw_ohlcv_{timeframe}",
    )
    overlap_periods = int(database_cfg.get("raw_overlap_periods", 5))
    insert_mode = str(database_cfg.get("raw_insert_mode", "replace")).lower()
    load_modifier = _resolve_load_modifier(insert_mode)
    timeframe_delta = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        `symbol` VARCHAR(32) NOT NULL,
        `timeframe` VARCHAR(16) NOT NULL,
        `timestamp` DATETIME NOT NULL,
        `open` DOUBLE NOT NULL,
        `high` DOUBLE NOT NULL,
        `low` DOUBLE NOT NULL,
        `close` DOUBLE NOT NULL,
        `volume` DOUBLE NOT NULL,
        PRIMARY KEY (`symbol`, `timeframe`, `timestamp`),
        KEY `idx_{table_name}_timestamp` (`timestamp`),
        KEY `idx_{table_name}_symbol_timestamp` (`symbol`, `timestamp`)
    ) ENGINE=InnoDB DEFAULT CHARSET={database_cfg.get("charset", "utf8mb4")};
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(create_sql)

        max_timestamp_map = _fetch_existing_raw_max_timestamps(connection, table_name, timeframe)
        total_rows_exported = 0

        for symbol, frame in asset_frames.items():
            export_frame = frame.copy()
            existing_max_timestamp = max_timestamp_map.get(symbol)
            if existing_max_timestamp is not None and not pd.isna(existing_max_timestamp):
                cutoff = existing_max_timestamp - (timeframe_delta * overlap_periods)
                export_frame = export_frame[export_frame.index >= cutoff]

            if export_frame.empty:
                LOGGER.info("Skipping raw OHLCV export for %s because no new rows were found.", symbol)
                continue

            export_count = _load_raw_frame_via_local_infile(
                connection=connection,
                table_name=table_name,
                symbol=symbol,
                timeframe=timeframe,
                frame=export_frame,
                load_modifier=load_modifier,
            )
            total_rows_exported += export_count
            LOGGER.info(
                "Exported %s raw OHLCV rows for %s into %s.",
                export_count,
                symbol,
                table_name,
            )

        LOGGER.info("Raw OHLCV export complete. %s rows processed into %s.", total_rows_exported, table_name)
    finally:
        connection.close()


def _build_sqlalchemy_engine(database_cfg: dict):
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError(
            "Database export requires SQLAlchemy. Install requirements.txt before enabling database export."
        ) from exc

    driver = str(database_cfg.get("driver", "mysql+pymysql"))
    host = database_cfg["host"]
    port = int(database_cfg.get("port", 3306))
    user = database_cfg["user"]
    password = database_cfg["password"]
    database = database_cfg["database"]
    charset = database_cfg.get("charset", "utf8mb4")

    url = f"{driver}://{user}:{password}@{host}:{port}/{database}?charset={charset}"
    return create_engine(url, future=True, pool_pre_ping=True, pool_recycle=3600)


def _build_pymysql_connection(database_cfg: dict):
    try:
        import pymysql
    except ImportError as exc:
        raise ImportError(
            "Raw database export requires PyMySQL. Install requirements.txt before enabling database export."
        ) from exc

    return pymysql.connect(
        host=database_cfg["host"],
        port=int(database_cfg.get("port", 3306)),
        user=database_cfg["user"],
        password=database_cfg["password"],
        database=database_cfg["database"],
        charset=database_cfg.get("charset", "utf8mb4"),
        local_infile=True,
        autocommit=True,
    )


def _prepare_frame_for_sql(frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        export_frame = frame.to_frame().reset_index()
    else:
        export_frame = frame.reset_index() if frame.index.name is not None or not isinstance(frame.index, pd.RangeIndex) else frame.copy()

    export_frame = export_frame.copy()
    for column in export_frame.columns:
        if pd.api.types.is_bool_dtype(export_frame[column]):
            export_frame[column] = export_frame[column].astype(int)
    return export_frame


def _build_table_name(prefix: str, logical_name: str) -> str:
    sanitized_name = "".join(character if character.isalnum() or character == "_" else "_" for character in logical_name.lower())
    return f"{prefix}_{sanitized_name}".strip("_")


def _resolve_load_modifier(insert_mode: str) -> str:
    if insert_mode == "replace":
        return "REPLACE"
    if insert_mode == "ignore":
        return "IGNORE"
    return ""


def _fetch_existing_raw_max_timestamps(connection, table_name: str, timeframe: str) -> Dict[str, pd.Timestamp]:
    query = f"""
    SELECT `symbol`, MAX(`timestamp`) AS max_timestamp
    FROM `{table_name}`
    WHERE `timeframe` = %s
    GROUP BY `symbol`
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (timeframe,))
        rows = cursor.fetchall()

    max_timestamp_map: Dict[str, pd.Timestamp] = {}
    for symbol, max_timestamp in rows:
        if max_timestamp is None:
            continue
        max_timestamp_map[str(symbol)] = pd.Timestamp(max_timestamp, tz="UTC")
    return max_timestamp_map


def _load_raw_frame_via_local_infile(
    connection,
    table_name: str,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
    load_modifier: str,
) -> int:
    export_frame = frame.reset_index().copy()
    export_frame.insert(0, "symbol", symbol)
    export_frame.insert(1, "timeframe", timeframe)
    export_frame["timestamp"] = pd.to_datetime(export_frame["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8", newline="") as handle:
        export_frame.to_csv(handle.name, index=False, header=False)
        temp_path = handle.name

    try:
        load_sql = f"""
        LOAD DATA LOCAL INFILE %s
        {load_modifier}
        INTO TABLE `{table_name}`
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\n'
        (`symbol`, `timeframe`, `timestamp`, `open`, `high`, `low`, `close`, `volume`)
        """
        with connection.cursor() as cursor:
            cursor.execute(load_sql, (temp_path,))
    finally:
        os.unlink(temp_path)

    return int(len(export_frame))
