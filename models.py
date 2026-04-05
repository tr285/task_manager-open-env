from pydantic import BaseModel
from typing import List

class Task(BaseModel):
    id: int
    priority: int
    deadline: int
    duration: int
    completed: bool

class Observation(BaseModel):
    time: int
    tasks: List[Task]
    completed: int
    missed: int

class Action(BaseModel):
    task_id: int