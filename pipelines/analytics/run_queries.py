import duckdb
from pathlib import Path
import re


# ============================================================
# Paths
# ============================================================

ANALYTICS_DIR = Path(__file__).resolve().parent
SQL_DIR = ANALYTICS_DIR / "sql"
OUTPUT_DIR = ANALYTICS_DIR / "output"


# ============================================================
# Extract queries from SQL file
# ============================================================

def extract_queries(sql_content):
    """
    Extract SQL queries based on numbered comments.

    Expected format:

        -- 1. Overall customer KPIs
        SELECT ...

        -- 2. Customer performance by state
        SELECT ...

        -- 3. Customers by number of orders
        SELECT ...
    """

    pattern = (
        r"--\s*(\d+)\.\s*(.*?)\n"
        r"(.*?)(?=\n\s*--\s*\d+\.\s*|\Z)"
    )

    matches = re.findall(
        pattern,
        sql_content,
        flags=re.DOTALL
    )

    queries = []

    for number, title, query in matches:

        # Convert title into a clean filename
        name = title.strip().lower()

        name = re.sub(
            r"[^a-z0-9]+",
            "_",
            name
        ).strip("_")

        queries.append(
            {
                "number": int(number),
                "name": name,
                "query": query.strip()
            }
        )

    return queries


# ============================================================
# Run one SQL file
# ============================================================

def run_sql_file(con, sql_file):

    print("\n" + "=" * 60)
    print(f"Processing: {sql_file.name}")
    print("=" * 60)

    sql_content = sql_file.read_text(
        encoding="utf-8"
    )

    queries = extract_queries(sql_content)

    if not queries:
        print(
            f"WARNING: No numbered queries found in "
            f"{sql_file.name}"
        )
        return False

    # Create output directory for this domain
    domain = sql_file.stem

    domain_output_dir = OUTPUT_DIR / domain

    domain_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_success = True

    # Execute every query
    for query_info in queries:

        query_number = query_info["number"]
        query_name = query_info["name"]
        query = query_info["query"]

        print(
            f"\nRunning query {query_number}: "
            f"{query_name}"
        )

        try:

            result = con.execute(query).fetchdf()

            output_file = (
                domain_output_dir
                / f"{query_name}.csv"
            )

            result.to_csv(
                output_file,
                index=False
            )

            print(
                f"  Rows returned: {len(result)}"
            )

            print(
                f"  Saved to: {output_file}"
            )

        except Exception as e:

            file_success = False

            print(
                f"  ERROR in query "
                f"{query_number}: {query_name}"
            )

            print(
                f"  {e}"
            )

    return file_success


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Starting Analytics Pipeline")
    print("=" * 60)

    # Check SQL directory
    if not SQL_DIR.exists():

        print(
            f"ERROR: SQL directory not found: "
            f"{SQL_DIR}"
        )

        raise SystemExit(1)

    # Find all SQL files
    sql_files = sorted(
        SQL_DIR.glob("*.sql")
    )

    if not sql_files:

        print("ERROR: No SQL files found.")

        raise SystemExit(1)

    print(
        f"\nFound {len(sql_files)} SQL files."
    )

    # Create DuckDB connection
    con = duckdb.connect()

    all_success = True

    try:

        # Execute every SQL file
        for sql_file in sql_files:

            success = run_sql_file(
                con,
                sql_file
            )

            if not success:
                all_success = False

    finally:

        # Always close connection
        con.close()

    print("\n" + "=" * 60)

    if all_success:

        print(
            "Analytics Pipeline Finished Successfully"
        )

    else:

        print(
            "Analytics Pipeline Finished "
            "with Errors"
        )

    print("=" * 60)

    if not all_success:
        raise SystemExit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()