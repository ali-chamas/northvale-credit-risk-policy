-- Loan status counts by issue year and term over the whole accepted-loans file.
-- Used to check censoring: what share of each vintage is still running at the snapshot.
-- Only three columns are read (DuckDB projection pushdown on the gzipped CSV).
SELECT
    CAST(substr(issue_d, 5, 4) AS INTEGER) AS issue_year,
    trim(term)                             AS term,
    loan_status,
    count(*)                               AS n_loans
FROM read_csv('{raw}', all_varchar = true, header = true)
WHERE issue_d IS NOT NULL
GROUP BY ALL
ORDER BY ALL
