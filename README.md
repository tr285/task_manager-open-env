---
title: Task Manager OpenEnv
emoji: ✅
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Task Manager OpenEnv

A simple task scheduling simulation environment deployed with Docker.

## Local Docker run

```bash
docker build -t task-manager-openenv .
docker run -p 7860:7860 task-manager-openenv
```

Then open `http://localhost:7860`

## Files

- `streamlit_app.py` → web UI
- `environment.py` → environment logic
- `models.py` → Pydantic models
- `tasks.py` → grading helpers

## Hugging Face Spaces

Create a new **Docker Space**, upload these files, and let the Space build.
