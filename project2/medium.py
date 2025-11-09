# medium.py — MountainCar (discretized) policy from offline data
# Book algorithms:
#   - Ch. 17.3: Q-learning target y = r + γ max_a' Q(s',a')
#   - Ch. 17.6: Linear action-value approximation; here we do Fitted Q-Iteration
#               via per-action ridge regression each epoch (least squares)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# ---- config ----
CSV_PATH = "data/medium.csv"
POLICY_OUT = "medium.policy"
NUM_STATES = 50000
NUM_ACTIONS = 7
GAMMA = 0.999      # small discount for stability (episodic but near-1)
EPOCHS = 12        # passes over data (can tweak)
RIDGE_LAMBDA = 1e-4  # tiny L2 to keep solves well-conditioned
SEED = 0

rng = np.random.default_rng(SEED)

# ---- helpers ----

def decode_pos_vel(s: int) -> Tuple[int, int]:
    """Decode (pos, vel) from 1-based state id."""
    s0 = s - 1
    pos = s0 % 500
    vel = s0 // 500
    return pos, vel

def build_base_features(states: np.ndarray) -> np.ndarray:
    """Base features on (pos, vel) scaled to [0,1]: [1, p, v, p*v, p^2, v^2]."""
    s0 = states - 1
    pos = s0 % 500
    vel = s0 // 500
    p = pos / 499.0
    v = vel / 99.0
    base = np.stack([
        np.ones_like(p, dtype=np.float64),
        p.astype(np.float64),
        v.astype(np.float64),
        (p*v).astype(np.float64),
        (p*p).astype(np.float64),
        (v*v).astype(np.float64),
    ], axis=1)  # (N,6)
    return base

def build_design_matrix(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Φ(s,a) = concat(base, one-hot(a))."""
    base = build_base_features(states)             # (N,6)
    oh = np.zeros((actions.shape[0], NUM_ACTIONS), dtype=np.float64)
    oh[np.arange(actions.shape[0]), actions - 1] = 1.0
    return np.concatenate([base, oh], axis=1)      # (N,13)

def q_values(states: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return Q(s,·) for a batch of states. W has one row per action."""
    base = build_base_features(states)             # (N,6)
    N = base.shape[0]
    Q = np.zeros((N, NUM_ACTIONS), dtype=np.float64)
    # build features per action quickly by appending that action's one-hot
    for a in range(1, NUM_ACTIONS+1):
        oh = np.zeros((N, NUM_ACTIONS), dtype=np.float64)
        oh[:, a-1] = 1.0
        Phi = np.concatenate([base, oh], axis=1)   # (N,13)
        Q[:, a-1] = Phi @ W[a-1]
    return Q

def greedy_action(states: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Greedy with lowest-index tie-break (np.argmax returns first max)."""
    Q = q_values(states, W)
    return 1 + np.argmax(Q, axis=1)

# ---- data + training ----

def load_medium(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"s","a","r","sp"}
    if not need.issubset(df.columns):
        raise ValueError("CSV must contain columns: s,a,r,sp")
    df["s"] = df["s"].astype(int)
    df["a"] = df["a"].astype(int)
    df["sp"] = df["sp"].astype(int)
    df["r"] = df["r"].astype(float)
    return df

def detect_terminals(df: pd.DataFrame) -> np.ndarray:
    """Treat states that appear as sp but never as s as terminals (data-driven)."""
    seen_s = set(df["s"].unique().tolist())
    seen_sp = set(df["sp"].unique().tolist())
    terminal = np.array(sorted(list(seen_sp - seen_s)), dtype=np.int64)
    return terminal

def fitted_q_iteration(df: pd.DataFrame, epochs=EPOCHS, gamma=GAMMA, lam=RIDGE_LAMBDA) -> np.ndarray:
    """
    FQI with linear Q:
      1) Build targets y = r + γ max_{a'} Q_W(sp, a')  (terminal -> y=r)
      2) For each action a, solve ridge: (Φ_a^T Φ_a + λI) θ_a = Φ_a^T y_a
    """
    # cache arrays
    s_arr  = df["s"].to_numpy(np.int64)
    a_arr  = df["a"].to_numpy(np.int64)
    r_arr  = df["r"].to_numpy(np.float64)
    sp_arr = df["sp"].to_numpy(np.int64)
    N = len(df)

    # terminal lookup for targets
    terminal_set = set(detect_terminals(df))

    # init weights: one vector per action (dim 13 = 6 base + 7 one-hot)
    D = 6 + NUM_ACTIONS
    W = np.zeros((NUM_ACTIONS, D), dtype=np.float64)

    # precompute per-action row indices to avoid recomputing masks each epoch
    idx_by_action = [np.where(a_arr == (a+1))[0] for a in range(NUM_ACTIONS)]

    for ep in range(epochs):
        # 1) targets with current W
        Q_next = q_values(sp_arr, W)               # (N,7)
        max_next = np.max(Q_next, axis=1)

        is_term = np.array([sp in terminal_set for sp in sp_arr], dtype=bool)
        y = r_arr.copy()
        y[~is_term] = r_arr[~is_term] + gamma * max_next[~is_term]

        # 2) per-action ridge regression
        for a in range(NUM_ACTIONS):
            rows = idx_by_action[a]
            if rows.size == 0:
                continue  # no samples for this action

            Phi_a = build_design_matrix(s_arr[rows], a_arr[rows])   # (n_a, 13)
            y_a   = y[rows]                                         # (n_a,)

            # normal equations with L2: (Φ^T Φ + λI) θ = Φ^T y
            AtA = Phi_a.T @ Phi_a
            # add tiny λ on diagonal
            AtA.flat[::AtA.shape[0]+1] += lam
            Aty = Phi_a.T @ y_a
            # solve for θ_a
            W[a] = np.linalg.solve(AtA, Aty)

        # uncomment to peek at training progress
        # print(f"epoch {ep+1}/{epochs} done")

    return W

def write_policy(W: np.ndarray, out_path: str):
    states = np.arange(1, NUM_STATES+1, dtype=np.int64)
    actions = greedy_action(states, W)
    with open(out_path, "w") as f:
        for a in actions:
            f.write(f"{int(a)}\n")

def main():
    if not Path(CSV_PATH).exists():
        print(f"Missing {CSV_PATH}. Put medium.csv under data/ and retry.")
        return
    df = load_medium(CSV_PATH)
    W = fitted_q_iteration(df, epochs=EPOCHS, gamma=GAMMA, lam=RIDGE_LAMBDA)
    write_policy(W, POLICY_OUT)
    print(f"Wrote policy to: {POLICY_OUT}")

if __name__ == "__main__":
    main()