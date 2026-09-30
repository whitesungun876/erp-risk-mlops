"""Read-only Databricks authentication and catalog probe.

Run only with a private, short-lived access-token file. Never print the token,
authenticated email, or a raw API response.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from databricks.sdk import WorkspaceClient


HOST = os.environ.get("DATABRICKS_HOST", "https://your-workspace.cloud.databricks.com")


def read_private_token(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("token file must not be a symlink")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise ValueError("token file must be a regular file readable only by its owner")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 20 or any(char.isspace() for char in token):
        raise ValueError("token file is empty or malformed")
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    client = WorkspaceClient(host=HOST, token=read_private_token(args.token_file))
    user = client.current_user.me()
    catalogs = list(client.catalogs.list())
    print(json.dumps({
        "authenticated": bool(user.id),
        "catalogs": sorted({catalog.name for catalog in catalogs if catalog.name}),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
