import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from environment import TaskManagerEnv
from models import Action

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "task-manager-openenv"
MAX_STEPS = 20


def _read_raw_env_token() -> Optional[str]:
    """Support malformed .env files that contain only the raw HF token."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return None

    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped and stripped.startswith("hf_"):
                return stripped
    except OSError:
        return None

    return None


def get_hf_token() -> Optional[str]:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or _read_raw_env_token()
    )


HF_TOKEN = get_hf_token()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)



def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )



def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )



def build_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        print("[WARN] No Hugging Face token found. Falling back to heuristic agent.", flush=True)
        return None

    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as exc:
        print(f"[WARN] Could not initialize OpenAI client: {exc}", flush=True)
        return None



def choose_task_heuristically(tasks) -> int:
    pending = [t for t in tasks if not t.completed]
    if not pending:
        return -1

    # Prefer urgent + high-priority + short tasks.
    pending.sort(key=lambda t: (t.deadline - t.priority, -t.priority, t.duration, t.id))
    return pending[0].id



def choose_task_with_llm(client: Optional[OpenAI], tasks) -> int:
    pending = [t for t in tasks if not t.completed]
    if not pending:
        return -1

    if client is None:
        return choose_task_heuristically(tasks)

    task_text = "\n".join(
        f"id={t.id}, priority={t.priority}, deadline={t.deadline}, duration={t.duration}"
        for t in pending
    )

    prompt = (
        "Choose the best next task to maximize reward.\n"
        "Prefer incomplete tasks with higher priority and urgent deadlines.\n"
        "Return only the task id as an integer.\n\n"
        f"Tasks:\n{task_text}"
    )

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only the task id as an integer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        text = (res.choices[0].message.content or "").strip()
        return int(text)
    except Exception as exc:
        print(f"[WARN] LLM selection failed, using heuristic agent instead: {exc}", flush=True)
        return choose_task_heuristically(tasks)



def main():
    client = build_client()
    env = TaskManagerEnv(difficulty=TASK_NAME)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):
            pending = [t for t in state.tasks if not t.completed]
            if not pending:
                success = True
                break

            task_id = choose_task_with_llm(client, state.tasks)
            if task_id == -1:
                break

            action = Action(task_id=task_id)

            try:
                state, reward, done, info = env.step(action)
                error = info.get("last_action_error") if isinstance(info, dict) else None
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            reward = max(0.0, min(float(reward), 1.0))
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"Action(task_id={task_id})",
                reward=reward,
                done=done,
                error=error,
            )

            if error:
                success = False
                break

            if done:
                success = all(t.completed for t in state.tasks)
                break

        if rewards:
            score = sum(rewards) / len(rewards)
        score = max(0.0, min(score, 1.0))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
