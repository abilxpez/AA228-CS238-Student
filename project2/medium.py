# medium.py — MountainCar (discretized) policy 
# Book:
#   • Ch. 17.3/17.6 target   : y = r + gamma * max_{a'} Q_theta(s_prime, a')   (terminal -> y = r)
#   • Ch. 17.6 representation: Q_theta(s, a) = theta^T * phi(s, a)
#   • Ch. 8 least squares    : (Phi^T Phi + lambda * I) * theta = Phi^T * y

import pandas as pd
import numpy as np
from typing import Tuple

CSV_PATH = "data/medium.csv"
POLICY_OUT = "medium.policy"
NUM_STATES = 50000
NUM_ACTIONS = 7
GAMMA = 0.999
EPOCHS = 12
RIDGE_LAMBDA = 1e-4
SEED = 0
rng = np.random.default_rng(SEED)

def decode_pos_vel(s: int) -> Tuple[int, int]:
    """map 1-based state id -> (pos_bin, vel_bin)."""
    # index mapping is 1 + pos + 500 * vel (project spec)
    # invert it (0-based) so s0 = s - 1; pos = s0 % 500; vel = s0 // 500
    s0 = s - 1
    pos = s0 % 500
    vel = s0 // 500
    return pos, vel

def build_base_features(states: np.ndarray) -> np.ndarray:
    # decode pos/vel bins from the indexing rule in the video
    s0 = states - 1
    pos = s0 % 500
    vel = s0 // 500

    # scale to [0,1] 
    p = pos / 499.0
    v = vel / 99.0

    # basic polynomial features on (p,v):
    # 1 (bias), p, v, p*v (interaction), p^2, v^2
    # small set that can capture curvature 
    base = np.stack([
        np.ones_like(p, dtype=np.float64),
        p.astype(np.float64),
        v.astype(np.float64),
        (p*v).astype(np.float64),
        (p*p).astype(np.float64),
        (v*v).astype(np.float64),
    ], axis=1)
    return base  # (N, 6)

def build_design_matrix(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    # build phi(s,a) by concatenating state features with an action one-hot
    base = build_base_features(states)  # (N, 6) from above
    one_hot = np.zeros((actions.shape[0], NUM_ACTIONS), dtype=np.float64)
    one_hot[np.arange(actions.shape[0]), actions - 1] = 1.0  # action indices are 1..7
    return np.concatenate([base, one_hot], axis=1)  # (N, 13) = 6 + 7

def q_values(states: np.ndarray, W: np.ndarray) -> np.ndarray:
    # compute Q_theta(s,a) = theta_a^T * phi(s,a) for every action
    base = build_base_features(states)
    N = base.shape[0]
    Q = np.zeros((N, NUM_ACTIONS), dtype=np.float64)

    # build phi(s, a=j) by appending the j-th one-hot
    for a in range(1, NUM_ACTIONS + 1):
        oh = np.zeros((N, NUM_ACTIONS), dtype=np.float64)
        oh[:, a-1] = 1.0
        Phi = np.concatenate([base, oh], axis=1)  # (N,13)
        Q[:, a-1] = Phi @ W[a-1]                  # linear Q with that action’s theta
    return Q  # (N, 7)

def greedy_action(states: np.ndarray, W: np.ndarray) -> np.ndarray:
    # pick the action with the largest Q(s,a)
    # np.argmax breaks ties by lowest index 
    Q = q_values(states, W)
    return 1 + np.argmax(Q, axis=1)

def load_medium(csv_path: str) -> pd.DataFrame:
    # load file and cast types 
    df = pd.read_csv(csv_path)
    df["s"]  = df["s"].astype(int)
    df["a"]  = df["a"].astype(int)
    df["sp"] = df["sp"].astype(int)
    df["r"]  = df["r"].astype(float)
    return df

def detect_terminals(df: pd.DataFrame) -> np.ndarray:
    # heuristic: terminal states appear as sp but never as s (no outgoing lines in data)
    seen_s = set(df["s"].unique().tolist())
    seen_sp = set(df["sp"].unique().tolist())
    return np.array(sorted(list(seen_sp - seen_s)), dtype=np.int64)

# implements the two main equations:
# 1) target: y = r + gamma * max_{a'} Q_theta(s', a')  (if s' terminal: y = r)
# 2) fit:    (Phi^T Phi + lam * I) * theta_a = Phi^T * y_a   for each action a
def fitted_q_iteration(df: pd.DataFrame, epochs=EPOCHS, gamma=GAMMA, lam=RIDGE_LAMBDA) -> np.ndarray:
    s_arr  = df["s"].to_numpy(np.int64)      # current states
    a_arr  = df["a"].to_numpy(np.int64)      # actions taken
    r_arr  = df["r"].to_numpy(np.float64)    # rewards
    sp_arr = df["sp"].to_numpy(np.int64)     # next states

    # precompute which next-states are terminal for the y = r branch
    terminal_set = set(detect_terminals(df))

    # one weight vector per action; dim = 6 state feats + 7 action one-hot
    D = 6 + NUM_ACTIONS
    W = np.zeros((NUM_ACTIONS, D), dtype=np.float64)

    # group dataset rows by action once so epochs are faster
    idx_by_action = [np.where(a_arr == (a+1))[0] for a in range(NUM_ACTIONS)]

    for ep in range(epochs):
        # build Q(s', ·) under current W to get the max term in the target
        Q_next = q_values(sp_arr, W)               # (N,7) values at next states
        max_next = np.max(Q_next, axis=1)          # elementwise max over actions

        # assemble targets y: copy r; then add gamma*max_next when s' is not terminal
        is_term = np.array([sp in terminal_set for sp in sp_arr], dtype=bool)
        y = r_arr.copy()
        y[~is_term] = r_arr[~is_term] + gamma * max_next[~is_term]

        # solve a small ridge system per action to refit theta_a to these targets
        for a in range(NUM_ACTIONS):
            rows = idx_by_action[a]
            if rows.size == 0:
                continue  # nothing in the dataset for this action

            # build Phi_a with the actual (s,a) pairs that used action a
            Phi_a = build_design_matrix(s_arr[rows], a_arr[rows])  # (n_a,13)
            y_a = y[rows]                                          # (n_a,)

            # normal equations with tiny L2: (Phi^T Phi + lam I) theta = Phi^T y
            AtA = Phi_a.T @ Phi_a
            AtA.flat[::AtA.shape[0]+1] += lam  # add lam on the diagonal
            Aty = Phi_a.T @ y_a
            W[a] = np.linalg.solve(AtA, Aty)   # update that action’s theta

        print(f"epoch {ep+1}/{epochs} done")
    return W

def write_policy(W: np.ndarray, out_path: str):
    # run greedy over all 50k states and print one action per line
    states = np.arange(1, NUM_STATES + 1, dtype=np.int64)
    actions = greedy_action(states, W)
    with open(out_path, "w") as f:
        for a in actions:
            f.write(f"{int(a)}\n")

def main():
    # train from CSV, write medium.policy
    df = load_medium(CSV_PATH)
    W = fitted_q_iteration(df, epochs=EPOCHS, gamma=GAMMA, lam=RIDGE_LAMBDA)
    write_policy(W, POLICY_OUT)
    print(f"Wrote policy to: {POLICY_OUT}")

if __name__ == "__main__":
    main()
