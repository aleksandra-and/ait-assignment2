from simple_grid import DrunkenWalkEnv
import numpy as np


def main(env, gamma=0.9):
    Q = np.zeros((env.nrow * env.ncol, 4))
    for iteration in range(1000):
        Q_new = np.zeros((env.nrow * env.ncol, 4))
        for s in env.P:
            if s == 12:  # don't update the value of the terminal state
                continue
            for a in env.P[s]:
                for (prob, next_state, reward, done) in env.P[s][a]:
                    max_next = max(Q[next_state])
                    Q_new[s][a] += prob * (reward + gamma * max_next)
        Q = Q_new

    print(Q)
    for i, row in enumerate(Q):
        maximum = max(row)
        # wonderful piece of code to convert the row into a latex table string
        row_string = ' & '.join(
            [str(i), *['{:.2e}'.format(q) if q != maximum else '\\textbf{' + '{:.2e}'.format(q) + '}' for q in row]])
        print(row_string + ' \\\\')


if __name__ == '__main__':
    main(DrunkenWalkEnv(map_name='theAlley'))


