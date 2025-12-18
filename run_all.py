import os
import signal
import subprocess
import sys
from typing import List


def build_commands() -> List[list]:
    python = sys.executable
    return [
        [python, "-m", "uvicorn", "src.api.server:app", "--reload", "--port", "8000"],
        [python, "-m", "chainlit", "run", "chainlit_app.py", "--watch"],
    ]


def terminate_processes(processes):
    for proc in processes:
        if proc.poll() is None:
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
    for cmd in commands:
        print(f"Starting: {' '.join(cmd)}")
        processes.append(subprocess.Popen(cmd))

    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
