"""Browser UI entry point: python app.py."""
import argparse

from vehicle_ai.web import create_app


def main():
    parser = argparse.ArgumentParser(description="Vehicle AI local web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
