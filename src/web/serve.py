"""Start the loopback-only workbench on an available local port."""

import argparse
import socket

import uvicorn

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    selected = None
    for port in range(args.port, args.port + 20):
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen(128)
            selected = (sock, port)
            break
        except OSError:
            sock.close()
    if selected is None:
        raise RuntimeError("No free workbench port")
    sock, port = selected
    print(f"Research workbench: http://127.0.0.1:{port}", flush=True)
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="info"))
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
