import os
import sys
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="AI Search Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    raw_port = os.environ.get("PORT", "8001")
    try:
        default_port = int(raw_port)
    except ValueError:
        default_port = 8001

    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI production server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind")
    serve_parser.add_argument("--port", type=int, default=default_port, help="Port to listen on")

    args = parser.parse_args()

    port = getattr(args, "port", default_port)
    host = getattr(args, "host", "0.0.0.0")

    print(f"Starting AI Search Engine on {host}:{port}...")
    from src.serving.app import app
    uvicorn.run(app, host=host, port=port, workers=1)

if __name__ == "__main__":
    main()

