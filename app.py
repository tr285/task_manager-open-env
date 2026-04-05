from flask import Flask, request, jsonify
from environment import TaskManagerEnv
from models import Action

app = Flask(__name__)

env = TaskManagerEnv()
current_state = env.reset()


def serialize_task(task):
    return {
        "id": task.id,
        "priority": task.priority,
        "deadline": task.deadline,
        "duration": task.duration,
        "completed": task.completed,
    }


def serialize_observation(obs):
    return {
        "time": obs.time,
        "tasks": [serialize_task(t) for t in obs.tasks],
        "completed": obs.completed,
        "missed": obs.missed,
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Task Manager OpenEnv API running"}), 200


@app.route("/reset", methods=["POST"])
def reset():
    global env, current_state
    data = request.get_json(silent=True) or {}
    difficulty = data.get("task", "easy")
    env = TaskManagerEnv(difficulty=difficulty)
    current_state = env.reset()
    return jsonify(serialize_observation(current_state)), 200


@app.route("/state", methods=["GET"])
def state():
    global current_state
    current_state = env.state()
    return jsonify(serialize_observation(current_state)), 200


@app.route("/step", methods=["POST"])
def step():
    global current_state
    data = request.get_json(silent=True) or {}

    if "task_id" not in data:
        return jsonify({"error": "task_id is required"}), 400

    action = Action(task_id=int(data["task_id"]))
    current_state, reward, done, info = env.step(action)

    return jsonify({
        "observation": serialize_observation(current_state),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)