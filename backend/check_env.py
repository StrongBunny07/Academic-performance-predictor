import os
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main() -> None:
    env_path = Path("backend/.env")
    _load_env_file(env_path)
    print("env file exists:", env_path.exists())
    print("GITHUB_TOKEN set:", bool(os.getenv("GITHUB_TOKEN")))
    print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    print("LLM_BASE_URL:", os.getenv("LLM_BASE_URL"))
    print("LLM_MODEL:", os.getenv("LLM_MODEL"))


if __name__ == "__main__":
    main()
