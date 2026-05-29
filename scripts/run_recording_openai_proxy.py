from __future__ import annotations

import argparse

from densify.openai_proxy.recording_proxy import make_proxy_server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8792)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--output-dir", default="runs/openai_recording_proxy")
    args = parser.parse_args()

    server = make_proxy_server(
        args.listen_host,
        args.listen_port,
        args.upstream_base_url,
        args.output_dir,
    )
    print(
        "recording proxy listening on "
        f"http://{args.listen_host}:{args.listen_port}/v1 -> {args.upstream_base_url}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
