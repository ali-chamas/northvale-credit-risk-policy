"""`make build`: scope the raw CSV with DuckDB and write typed parquet to data/interim.

Steps
1. Status counts by issue year and term over the whole file (censoring check).
2. 36-month loans issued 2012-01 to 2015-12, all 151 columns as text.
3. Cast types: a column becomes numeric when every non-empty value parses as a
   number; month fields ("Dec-2015") become dates. Add target and split.

The 2.2M-row file is never loaded into pandas: DuckDB streams it, reads only the
columns a query needs and pushes the filters down.
"""

from __future__ import annotations

import time

import duckdb

from . import config as C

DATE_COLUMNS = [
    "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
    "sec_app_earliest_cr_line", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "debt_settlement_flag_date", "settlement_date",
]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")
    con.execute(f"SET threads = {C.N_JOBS}")
    con.execute(f"SET temp_directory = '{C.INTERIM / 'duckdb_tmp'}'")
    return con


def sql(name: str, **kw) -> str:
    return (C.SQL / name).read_text().format(**kw)


def status_by_vintage(con) -> None:
    df = con.execute(sql("01_status_by_vintage.sql", raw=C.RAW_FILE)).df()
    total = int(df["n_loans"].sum())
    if total != C.RAW_ROWS:
        raise ValueError(f"Expected {C.RAW_ROWS:,} loans in the raw file, found {total:,}")
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES / "status_by_vintage_all_terms.csv", index=False)
    print(f"build: {total:,} loans in raw file; status table written")


def scope(con) -> None:
    con.execute(sql("02_scope.sql", raw=C.RAW_FILE, out=C.SCOPE_TEXT))
    n = con.execute(f"SELECT count(*) FROM '{C.SCOPE_TEXT}'").fetchone()[0]
    print(f"build: {n:,} 36-month loans issued 2012-2015")


def typed(con) -> None:
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{C.SCOPE_TEXT}'").fetchall()]
    # A column is numeric when every non-empty value casts to DOUBLE.
    probe = ", ".join(
        f"count(NULLIF(trim(\"{c}\"), '')) = count(TRY_CAST(NULLIF(trim(\"{c}\"), '') AS DOUBLE)) AS \"{c}\""
        for c in cols
    )
    is_num = con.execute(f"SELECT {probe} FROM '{C.SCOPE_TEXT}'").df().iloc[0].to_dict()
    select = []
    for c in cols:
        if c in DATE_COLUMNS:
            select.append(f"CAST(try_strptime(NULLIF(trim(\"{c}\"), ''), '%b-%Y') AS DATE) AS \"{c}\"")
        elif is_num[c] and c not in ("id", "member_id", "zip_code", "policy_code"):
            select.append(f"TRY_CAST(NULLIF(trim(\"{c}\"), '') AS DOUBLE) AS \"{c}\"")
        else:
            select.append(f"NULLIF(trim(\"{c}\"), '') AS \"{c}\"")
    bad = ", ".join(f"'{s}'" for s in C.BAD_STATUSES)
    good = ", ".join(f"'{s}'" for s in C.GOOD_STATUSES)
    split = " ".join(
        f"WHEN issue_d BETWEEN DATE '{a}' AND DATE '{b}' THEN '{k}'" for k, (a, b) in C.SPLITS.items()
    )
    con.execute(
        f"""
        COPY (
            WITH t AS (SELECT {", ".join(select)} FROM '{C.SCOPE_TEXT}')
            SELECT *,
                CASE WHEN loan_status IN ({bad}) THEN 1
                     WHEN loan_status IN ({good}) THEN 0 END AS target,
                CASE {split} END AS split,
                (application_type = 'Individual' AND loan_status IN ({bad}, {good})) AS in_sample
            FROM t
            ORDER BY issue_d, id
        ) TO '{C.LOANS}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    summary = con.execute(
        f"SELECT split, count(*) n, sum(in_sample::INT) n_sample, avg(target) bad_rate FROM '{C.LOANS}' GROUP BY 1 ORDER BY 1"
    ).df()
    print(summary.to_string(index=False))


def main() -> None:
    t0 = time.time()
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    con = connect()
    status_by_vintage(con)
    scope(con)
    typed(con)
    print(f"build: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
