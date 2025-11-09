# large.py — offline planning for the secret large MDP
# Book: Ch. 16 (MLE model from data) + Value Iteration (plain or prioritized)

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import heapq
from pathlib import Path
from typing import Dict, Tuple

# ---- defaults (can override via CLI) ----
NUM_STATES = 302020
NUM_ACTIONS = 9
GAMMA_DEFAULT = 0.95
TOL_DEFAULT = 1e-6
MAX_SWEEPS_DEFAULT = 2000
MAX_POPS_DEFAULT = 3_000_000  # prioritized mode safety cap
UNSEEN_DEFAULT = "zero"       # options: "zero", "self_loop"
UNSEEN_REWARD_DEFAULT = 0.0   # reward to use when unseen + self_loop

# ---------- IO ----------

def load_transitions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expect s,a,r,sp
    need = {"s", "a", "r", "sp"}
    if not need.issubset(df.columns):
        raise ValueError("CSV must contain columns: s,a,r,sp")
    # keep types tight
    df["s"] = df["s"].astype(np.int64)
    df["a"] = df["a"].astype(np.int64)
    df["sp"] = df["sp"].astype(np.int64)
    df["r"] = df["r"].astype(np.float32)
    return df

# ---------- MLE model (Ch.16) ----------

def mle_estimate(df: pd.DataFrame, S: int, A: int):
    """
    Build MLE estimates:
      T_hat(s'|s,a) from counts; R_hat(s,a) mean reward.
    We store transitions sparsely: (s,a) -> (np.array sps, np.array probs).
    """
    counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    reward_sum: Dict[Tuple[int, int], float] = defaultdict(float)

    # one pass over the data
    for s, a, r, sp in df[["s", "a", "r", "sp"]].itertuples(index=False, name=None):
        counts[(s, a)][sp] += 1
        reward_sum[(s, a)] += float(r)

    # mean reward table (1-based for convenience)
    R = np.zeros((S + 1, A + 1), dtype=np.float32)
    # sparse transitions
    T_sparse: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    for (s, a), cnts in counts.items():
        n = float(sum(cnts.values()))
        R[s, a] = reward_sum[(s, a)] / n if n > 0 else 0.0
        sps = np.fromiter(cnts.keys(), dtype=np.int32)
        probs = np.fromiter((c / n for c in cnts.values()), dtype=np.float32)
        probs = probs / probs.sum()  # tiny numeric guard
        T_sparse[(s, a)] = (sps, probs)

    # build predecessor map: for prioritized updates we need Pred(s') = {p | ∃a: p --a--> s'}
    preds: Dict[int, list] = defaultdict(list)
    for (p, a), (sps, probs) in T_sparse.items():
        # only record predecessors that actually have nonzero prob
        for sp in sps:
            preds[int(sp)].append(int(p))

    return R, T_sparse, preds

# ---------- Bellman helpers ----------

def q_value_for(R, T_sparse, V, s: int, a: int, gamma: float, unseen_mode: str, unseen_reward: float):
    """
    Q(s,a) = R(s,a) + gamma * sum_{s'} T(s'|s,a) V[s'].
    Unseen handling:
      - "zero": if (s,a) not seen, return 0.0 (strict ML convention).
      - "self_loop": treat as P(s|s,a)=1 with reward 'unseen_reward'.
    """
    key = (s, a)
    if key in T_sparse:
        sps, probs = T_sparse[key]
        return float(R[s, a]) + gamma * float(np.dot(probs, V[sps]))
    if unseen_mode == "self_loop":
        return unseen_reward + gamma * float(V[s])
    # default: zero
    return 0.0

def bellman_backup(R, T_sparse, V, s: int, gamma: float, unseen_mode: str, unseen_reward: float) -> float:
    """One-step backup at state s: max over actions; lowest index wins on ties."""
    best_q = -np.inf
    best_a = 1
    for a in range(1, NUM_ACTIONS + 1):
        q = q_value_for(R, T_sparse, V, s, a, gamma, unseen_mode, unseen_reward)
        if q > best_q:
            best_q = q
            best_a = a
    # if we never saw anything at s and unseen_mode is "zero", best_q could be 0.0 (fine)
    return best_q

# ---------- Value Iteration (two modes) ----------

def value_iteration_plain(R, T_sparse, S: int, gamma: float, tol: float,
                          max_sweeps: int, unseen_mode: str, unseen_reward: float):
    """Standard Gauss–Seidel sweeps."""
    V = np.zeros(S + 1, dtype=np.float64)
    for sweep in range(max_sweeps):
        delta = 0.0
        for s in range(1, S + 1):
            new_v = bellman_backup(R, T_sparse, V, s, gamma, unseen_mode, unseen_reward)
            d = abs(new_v - V[s])
            if d > delta:
                delta = d
            V[s] = new_v
        print(f"sweep {sweep+1}  delta={delta:.3e}")
        if delta < tol:
            break
    return V

def value_iteration_prioritized(R, T_sparse, preds, S: int, gamma: float, tol: float,
                                max_pops: int, unseen_mode: str, unseen_reward: float):
    """
    Prioritized sweeping:
      - keep a max-heap of residuals
      - update the top state, then push its predecessors.
    """
    V = np.zeros(S + 1, dtype=np.float64)

    # initial residuals (compute only for states that appear as sources or have preds)
    touched = set([s for (s, _) in T_sparse.keys()]) | set(preds.keys())
    heap = []
    for s in touched:
        new_v = bellman_backup(R, T_sparse, V, s, gamma, unseen_mode, unseen_reward)
        res = abs(new_v - V[s])
        if res > 0.0:
            heapq.heappush(heap, (-res, int(s)))

    pops = 0
    last_report = 0
    while heap and pops < max_pops:
        neg_res, s = heapq.heappop(heap)
        res = -neg_res
        # if the residual is already small, we can stop early
        if res < tol:
            break

        # update V[s]
        new_v = bellman_backup(R, T_sparse, V, s, gamma, unseen_mode, unseen_reward)
        V[s] = new_v
        pops += 1

        # propagate to predecessors of s
        for p in preds.get(s, []):
            new_v_p = bellman_backup(R, T_sparse, V, p, gamma, unseen_mode, unseen_reward)
            res_p = abs(new_v_p - V[p])
            if res_p > tol:
                heapq.heappush(heap, (-res_p, p))

        # lightweight progress print
        if pops - last_report >= 200000:
            print(f"pops={pops}  current_res={res:.3e}  heap={len(heap)}")
            last_report = pops

    print(f"prioritized finished: pops={pops}, heap_left={len(heap)}")
    return V

# ---------- Policy extraction + save ----------

def extract_policy(R, T_sparse, V, S: int, gamma: float, unseen_mode: str, unseen_reward: float) -> np.ndarray:
    """Greedy deterministic policy; lowest action index on ties."""
    policy = np.ones(S + 1, dtype=np.int32)
    for s in range(1, S + 1):
        best_q = -np.inf
        best_a = 1
        for a in range(1, NUM_ACTIONS + 1):
            q = q_value_for(R, T_sparse, V, s, a, gamma, unseen_mode, unseen_reward)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return policy

def save_policy(policy: np.ndarray, out_path: str, S: int):
    with open(out_path, "w") as f:
        for s in range(1, S + 1):
            f.write(f"{int(policy[s])}\n")

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Large MDP planner (Ch.16 MLE + VI)")
    p.add_argument("--csv", type=str, default="data/large.csv", help="path to large.csv")
    p.add_argument("--out", type=str, default="large.policy", help="output policy file")
    p.add_argument("--gamma", type=float, default=GAMMA_DEFAULT, help="discount factor")
    p.add_argument("--tol", type=float, default=TOL_DEFAULT, help="VI tolerance")
    p.add_argument("--mode", type=str, default="prioritized", choices=["prioritized", "plain"],
                   help="VI mode")
    p.add_argument("--max_sweeps", type=int, default=MAX_SWEEPS_DEFAULT, help="max sweeps (plain VI)")
    p.add_argument("--max_pops", type=int, default=MAX_POPS_DEFAULT, help="max heap pops (prioritized)")
    p.add_argument("--unseen", type=str, default=UNSEEN_DEFAULT, choices=["zero", "self_loop"],
                   help="how to handle unseen (s,a)")
    p.add_argument("--unseen_reward", type=float, default=UNSEEN_REWARD_DEFAULT,
                   help="reward used when unseen='self_loop'")
    return p.parse_args()

# ---------- main ----------

def main():
    args = parse_args()
    if not Path(args.csv).exists():
        print(f"Missing {args.csv}. Put large.csv under data/ and retry.")
        return

    print("loading data...")
    df = load_transitions(args.csv)

    print("building MLE model (Ch.16)...")
    R, T_sparse, preds = mle_estimate(df, S=NUM_STATES, A=NUM_ACTIONS)

    print(f"planning with Value Iteration ({args.mode})...")
    if args.mode == "plain":
        V = value_iteration_plain(
            R, T_sparse, S=NUM_STATES,
            gamma=args.gamma, tol=args.tol, max_sweeps=args.max_sweeps,
            unseen_mode=args.unseen, unseen_reward=args.unseen_reward
        )
    else:
        V = value_iteration_prioritized(
            R, T_sparse, preds, S=NUM_STATES,
            gamma=args.gamma, tol=args.tol, max_pops=args.max_pops,
            unseen_mode=args.unseen, unseen_reward=args.unseen_reward
        )

    print("extracting policy...")
    policy = extract_policy(
        R, T_sparse, V, S=NUM_STATES,
        gamma=args.gamma, unseen_mode=args.unseen, unseen_reward=args.unseen_reward
    )
    save_policy(policy, args.out, S=NUM_STATES)
    print(f"wrote policy to: {args.out}")

if __name__ == "__main__":
    main()
