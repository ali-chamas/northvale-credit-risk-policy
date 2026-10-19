-- Scope: 36-month loans issued January 2012 to December 2015, every raw column kept
-- (as text) so the feature audit can cover all 151 columns. Types are cast in the
-- next step. Rows with no issue date are the summary lines at the end of the file.
COPY (
    SELECT *
    FROM read_csv('{raw}', all_varchar = true, header = true)
    WHERE issue_d IS NOT NULL
      AND trim(term) = '36 months'
      AND strptime(issue_d, '%b-%Y') BETWEEN DATE '2012-01-01' AND DATE '2015-12-01'
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
