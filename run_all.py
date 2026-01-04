import os
import signal
import subprocess
import sys
import time
from typing import List


def build_commands() -> List[list]:
    python = sys.executable
    chainlit_port = os.getenv("CHAINLIT_PORT", "8001")
    host = os.getenv("HOST", "127.0.0.1")
    return [
        [
            python,
            "-m",
            "uvicorn",
            "src.api.server:app",
            "--reload",
            "--host",
            host,
            "--port",
            "8000",
        ],
        [
            python,
            "-m",
            "chainlit",
            "run",
            "chainlit_app.py",
            "--watch",
            "--host",
            host,
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
