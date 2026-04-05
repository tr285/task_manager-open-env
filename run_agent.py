import random
from environment import TaskManagerEnv
from models import Action

random.seed(42)

env = TaskManagerEnv()
state = env.reset()

total_reward = 0

while True:
    tasks = state.tasks

    best = None
    max_p = -1

    for t in tasks:
        if not t.completed:
            if t.priority > max_p:
                max_p = t.priority
                best = t

    if best is None:
        break

    action = Action(task_id=best.id)

    state, reward, done, _ = env.step(action)
    total_reward += reward

    if done:
        break

print("Baseline Reward:", total_reward)