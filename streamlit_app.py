import random
import streamlit as st
from environment import TaskManagerEnv
from models import Action
from tasks import grade_easy, grade_medium, grade_hard

st.set_page_config(page_title='Task Manager Env', page_icon='✅', layout='wide')


def init_env(seed: int = 42):
    random.seed(seed)
    env = TaskManagerEnv()
    state = env.reset()
    return env, state


if 'env' not in st.session_state:
    st.session_state.env, st.session_state.state = init_env()
    st.session_state.total_reward = 0
    st.session_state.last_reward = 0
    st.session_state.done = False


def reset_app():
    st.session_state.env, st.session_state.state = init_env()
    st.session_state.total_reward = 0
    st.session_state.last_reward = 0
    st.session_state.done = False


def step_task(task_id: int):
    if st.session_state.done:
        return
    action = Action(task_id=task_id)
    state, reward, done, _ = st.session_state.env.step(action)
    st.session_state.state = state
    st.session_state.last_reward = reward
    st.session_state.total_reward += reward
    st.session_state.done = done


def run_best_priority_agent():
    while not st.session_state.done:
        tasks = st.session_state.state.tasks
        candidates = [t for t in tasks if not t.completed]
        if not candidates:
            break
        best = max(candidates, key=lambda t: t.priority)
        step_task(best.id)


st.title('Task Manager OpenEnv')
st.caption('Simple task scheduling simulation with manual controls and an auto-play baseline agent.')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Time', st.session_state.state.time)
col2.metric('Completed', st.session_state.state.completed)
col3.metric('Missed', st.session_state.state.missed)
col4.metric('Total Reward', st.session_state.total_reward)

st.write(f'Last reward: **{st.session_state.last_reward}**')

left, right = st.columns([2, 1])

with left:
    st.subheader('Tasks')
    tasks = st.session_state.state.tasks
    table_data = []
    for t in tasks:
        status = 'Done' if t.completed else 'Pending'
        table_data.append({
            'id': t.id,
            'priority': t.priority,
            'deadline': t.deadline,
            'duration_left': max(t.duration, 0),
            'status': status,
        })
    st.dataframe(table_data, use_container_width=True)

with right:
    st.subheader('Actions')
    pending = [t for t in st.session_state.state.tasks if not t.completed]
    if pending and not st.session_state.done:
        chosen = st.selectbox('Choose task id', [t.id for t in pending])
        if st.button('Do one step', use_container_width=True):
            step_task(chosen)
    else:
        st.info('No pending tasks left or episode finished.')

    if st.button('Run best-priority agent', use_container_width=True):
        run_best_priority_agent()

    if st.button('Reset', use_container_width=True):
        reset_app()

st.subheader('Scores')
env = st.session_state.env
c1, c2, c3 = st.columns(3)
c1.metric('Easy', round(grade_easy(env), 3))
c2.metric('Medium', round(grade_medium(env), 3))
c3.metric('Hard', round(grade_hard(env), 3))

if st.session_state.done:
    st.success('Episode finished.')
