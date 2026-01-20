import os
import signal
import subprocess
import sys
import time
from typing import List


def _load_env_file() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (
                    len(value) >= 2
                    and value[0] == value[-1]
                    and value[0] in {"'", '"'}
                ):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    except Exception:
        return


def build_commands() -> List[list]:
    python = sys.executable
    chainlit_port = os.getenv("CHAINLIT_PORT", "8001")
    fastapi_port = os.getenv("FASTAPI_PORT", "8000")
    fastapi_host = os.getenv("FASTAPI_HOST", os.getenv("HOST", "0.0.0.0"))
    chainlit_host = os.getenv("CHAINLIT_HOST", os.getenv("HOST", "127.0.0.1"))
    return [
        [
            python,
            "-m",
            "uvicorn",
            "src.api.server:app",
            "--host",
            fastapi_host,
            "--port",
            fastapi_port,
        ],
        [
            python,
            "-m",
            "chainlit",
            "run",
            "chainlit_app.py",
            "--host",
            chainlit_host,
            "--port",
            chainlit_port,
        ],
    ]


def terminate_processes(processes):
    alive = [proc for proc in processes if proc.poll() is None]
    if not alive:
        return

    if os.name == "nt":
        for proc in alive:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
        deadline = time.time() + 3
        for proc in alive:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        for proc in alive:
            if proc.poll() is None:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except Exception:
                    proc.kill()
    else:
        for proc in alive:
            proc.terminate()

    for proc in processes:
        if proc.poll() is None:
            proc.wait()


def main():
    processes: List[subprocess.Popen] = []
    _load_env_file()

    def handle_signal(signum, frame):
        terminate_processes(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    commands = build_commands()
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    for cmd in commands:
        print(f"Starting: {' '.join(cmd)}")
        processes.append(subprocess.Popen(cmd, creationflags=creationflags))

    try:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    terminate_processes(processes)
                    return
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
