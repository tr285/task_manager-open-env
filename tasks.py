def grade_easy(env):
    return env.completed / len(env.tasks)


def grade_medium(env):
    total = 0
    done = 0

    for t in env.tasks:
        total += t.priority
        if t.completed:
            done += t.priority

    return done / total


def grade_hard(env):
    score = env.completed - env.missed

    if score < 0:
        score = 0

    return score / len(env.tasks)