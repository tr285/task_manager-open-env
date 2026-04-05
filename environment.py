import random
from models import Task, Observation, Action


class TaskManagerEnv:
    def __init__(self, difficulty: str = "easy", seed: int | None = 42):
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.max_time = 20
        self.max_tasks = 5
        self.reset()

    def _generate_tasks(self):
        tasks = []
        for i in range(1, self.max_tasks + 1):
            if self.difficulty == "easy":
                priority = self.rng.randint(3, 5)
                deadline = self.rng.randint(8, 15)
                duration = self.rng.randint(1, 2)
            elif self.difficulty == "medium":
                priority = self.rng.randint(2, 5)
                deadline = self.rng.randint(6, 12)
                duration = self.rng.randint(1, 3)
            else:
                priority = self.rng.randint(1, 5)
                deadline = self.rng.randint(5, 10)
                duration = self.rng.randint(2, 4)

            tasks.append(
                Task(
                    id=i,
                    priority=priority,
                    deadline=deadline,
                    duration=duration,
                    completed=False,
                )
            )
        return tasks

    def reset(self):
        self.time = 0
        self.tasks = self._generate_tasks()
        self.completed = 0
        self.missed = 0
        return self.state()

    def state(self):
        return Observation(
            time=self.time,
            tasks=self.tasks,
            completed=self.completed,
            missed=self.missed,
        )

    def step(self, action: Action):
        reward = 0.0
        info = {"last_action_error": None}

        task = next((t for t in self.tasks if t.id == action.task_id), None)

        if task is None:
            info["last_action_error"] = "invalid task_id"
            return self.state(), 0.0, False, info

        if task.completed:
            info["last_action_error"] = "task already completed"
            return self.state(), 0.0, False, info

        task.duration -= 1
        reward += 0.2

        if task.duration <= 0:
            task.completed = True
            if self.time <= task.deadline:
                self.completed += 1
                reward += min(task.priority * 0.2, 0.8)
            else:
                self.missed += 1
                reward += 0.0

        for t in self.tasks:
            if not t.completed and self.time > t.deadline:
                t.completed = True
                self.missed += 1

        self.time += 1

        done = self.time >= self.max_time or all(t.completed for t in self.tasks)

        reward = max(0.0, min(float(reward), 1.0))

        return self.state(), reward, done, info