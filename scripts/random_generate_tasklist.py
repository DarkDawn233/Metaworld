import random
from copy import deepcopy
from metaworld.envs.display_utils import (
    TASKS, STATES, PRECONDITIONS, check_task_cond, change_state)


TOTAL_TASKS = deepcopy(list(PRECONDITIONS.keys()))


def init_states() -> dict[str, str]:
    states = {'cup': STATES.CUP_STATE_DESK,
              'drawer': STATES.DRAWER_STATE_CLOSED}
    return states


def simulate(states, num_tasks: int = 20) -> dict[str, int]:
    task_count = {t: 0 for t in TOTAL_TASKS}
    for _ in range(num_tasks):
        valid_tasks = list()
        for next_cand_task in TOTAL_TASKS:
            if check_task_cond(next_cand_task, states):
                valid_tasks.append(next_cand_task)
        task = random.choice(valid_tasks)
        states = change_state(task, states)
        task_count[task] += 1
    return task_count


def random_sample_tasklist(num_trials: int = 1e5):
    num_trials = int(num_trials)
    task_count = {t: 0 for t in TOTAL_TASKS}
    for _ in range(num_trials):
        states = init_states()
        task_count_one_trial = simulate(states)
        for task in TOTAL_TASKS:
            task_count[task] += task_count_one_trial[task]
        print(task_count)
    probs = {k: v/num_trials/20 for k, v in task_count.items()}
    print(task_count)
    print(probs)


if __name__ == '__main__':
    random_sample_tasklist()
