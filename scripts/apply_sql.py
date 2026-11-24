"""Apply the SQL migrations in order to a Postgres database.

Usage:
    DATABASE_URL=postgresql://... python scripts/apply_sql.py [--with-shim]

--with-shim also applies sql/dev/000_local_shim.sql first. Use it for plain
Postgres (local or CI). Never use it against a Supabase project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def migration_files(with_shim: bool) -> list[Path]:
    files = sorted(SQL_DIR.glob("0*.sql"))
    if with_shim:
        files = sorted((SQL_DIR / "dev").glob("0*.sql")) + files
    return files


def main(argv: list[str]) -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2
    try:
        import psycopg
    except ImportError:
        print("psycopg is required: pip install 'psycopg[binary]'", file=sys.stderr)
        return 2
    with psycopg.connect(url, autocommit=True) as conn:
        for path in migration_files("--with-shim" in argv):
            conn.execute(path.read_text(encoding="utf-8"))
            print(f"applied {path.relative_to(SQL_DIR.parent).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
