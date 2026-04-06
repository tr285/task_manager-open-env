from fastapi import FastAPI
from environment import TaskManagerEnv
from models import Action

app = FastAPI()

env = TaskManagerEnv()

@app.get("/")
def home():
    return {"message": "Task Manager OpenEnv API running"}

@app.post("/reset")
def reset():
    state = env.reset()
    return state.dict()

@app.post("/step")
def step(action: Action):
    state, reward, done, _ = env.step(action)
    return {
        "state": state.dict(),
        "reward": reward,
        "done": done
    }

@app.get("/state")
def get_state():
    return env.state().dict()