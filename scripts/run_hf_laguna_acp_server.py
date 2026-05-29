from __future__ import annotations

import argparse

from densify.acp.server import run_dummy_server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dummy"], default="dummy")
    parser.add_argument("--log-path", default="runs/acp_spike/messages.jsonl")
    parser.add_argument("--fixed-response", default="READY")
    args = parser.parse_args()

    if args.mode == "dummy":
        run_dummy_server(args.log_path, args.fixed_response)


if __name__ == "__main__":
    main()

