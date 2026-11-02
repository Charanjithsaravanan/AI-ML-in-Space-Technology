"""Run the GEOSAT API and bundled dashboard as one local application."""
from __future__ import annotations

import os
import webbrowser

import uvicorn

from api.app import app


def main() -> None:
    host = os.getenv("GEOSAT_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    url = f"http://{host}:{port}"
    webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
