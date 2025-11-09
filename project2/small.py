import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Tuple

CSV_PATH = "data/small.csv" 
POLICY_OUT = "small.policy"
NUM_STATES = 100
NUM_ACTIONS = 4
GAMMA = 0.95
TOL = 1e-6
MAX_SWEEPS = 10000  # safety cap

def load_transitions(csv_path: str):
    df = pd.read_csv(csv_path)
    # expected columns
    if not set(["s","a","r","sp"]).issubset(df.columns):
        raise ValueError("CSV must contain columns: s,a,r,sp")
    return df

# Book Ch.16 (MLE):
#   T(s'|s,a) = N(s,a,s') / N(s,a)
#   R(s,a) = sum_r(s,a) / N(s,a)
# Unseen (s,a): contributes 0 in backups (no smoothing).
def mle_estimate(df: pd.DataFrame):
    # keep counts of next states for each (s,a)
    counts: Dict[Tuple[int,int], Counter] = defaultdict(Counter)
    # sum rewards for each (s,a) to average later
    reward_sum: Dict[Tuple[int,int], float] = defaultdict(float)

    # one pass over the dataset to fill counts and reward sums
    for s, a, r, sp in df[["s","a","r","sp"]].itertuples(index=False, name=None):
        counts[(s,a)][sp] += 1
        reward_sum[(s,a)] += float(r)

    # allocate mean rewards and sparse transition dict
    R = np.zeros((NUM_STATES+1, NUM_ACTIONS+1), dtype=np.float64)  # 1-based
    # map (s,a) -> (array of next states, array of probs)
    T_sparse: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray]] = {}

    # convert counts to probabilities and compute average rewards
    for (s,a), cnts in counts.items():
        n = float(sum(cnts.values()))              
        R[s,a] = reward_sum[(s,a)] / n if n > 0 else 0.0

        # turn the keys and values of the counter into numpy arrays
        sps = np.fromiter(cnts.keys(), dtype=np.int64)
        probs = np.fromiter((c/n for c in cnts.values()), dtype=np.float64)
        probs = probs / probs.sum()  # small numeric guard
        T_sparse[(s,a)] = (sps, probs)

    return R, T_sparse

# Book Ch.16 Value Iteration (Gauss–Seidel):
#   V(s) ← max_a [ R(s,a) + γ * Σ_{s'} T(s'|s,a) V(s') ]
# Unseen (s,a): Q(s,a)=0.
def value_iteration(R, T_sparse, gamma=GAMMA, tol=TOL, max_sweeps=MAX_SWEEPS):
    # start with zero values; index 0 unused to match 1..NUM_STATES
    V = np.zeros(NUM_STATES+1, dtype=np.float64)
    sweeps = 0

    while sweeps < max_sweeps:
        sweeps += 1
        delta = 0.0  # track largest change this sweep

        # Gauss–Seidel: update V[s] in place as we go left→right
        for s in range(1, NUM_STATES+1):
            best_q = -np.inf

            # compute Q(s,a) for each action
            for a in range(1, NUM_ACTIONS+1):
                if (s,a) in T_sparse:
                    sps, probs = T_sparse[(s,a)]
                    # Bellman one-step lookahead for this action
                    q = R[s,a] + gamma * np.dot(probs, V[sps])
                else:
                    # no samples for (s,a) → treat as 0 contribution
                    q = 0.0

                # keep the max Q over actions
                if q > best_q:
                    best_q = q

            # if we never saw any (s,a), keep V[s]=0
            new_vs = best_q if best_q != -np.inf else 0.0
            delta = max(delta, abs(new_vs - V[s]))
            V[s] = new_vs

        # stop when values stop changing much
        if delta < tol:
            break

    return V, sweeps

# Greedy policy with lowest-index tie-break.
def extract_policy(V, R, T_sparse):
    # default to action 1 so we always output something
    policy = np.ones(NUM_STATES+1, dtype=np.int64)

    for s in range(1, NUM_STATES+1):
        best_a = 1
        best_q = -np.inf

        # score each action using the final V
        for a in range(1, NUM_ACTIONS+1):
            if (s,a) in T_sparse:
                sps, probs = T_sparse[(s,a)]
                q = R[s,a] + GAMMA * np.dot(probs, V[sps])
            else:
                q = 0.0

            # strict ">" keeps lower action id on ties
            if q > best_q:
                best_q = q
                best_a = a

        policy[s] = best_a

    return policy

def save_policy(policy, path):
    with open(path, "w") as f:
        for s in range(1, NUM_STATES+1):
            f.write(f"{policy[s]}\n")

# ---- run for small.csv ----
df_small = load_transitions(CSV_PATH)
R_hat, T_hat_sparse = mle_estimate(df_small)

V, sweeps = value_iteration(R_hat, T_hat_sparse, gamma=GAMMA, tol=TOL)
policy = extract_policy(V, R_hat, T_hat_sparse)
save_policy(policy, POLICY_OUT)

print(f"Value Iteration converged in {sweeps} sweeps with tol={TOL}")
print(f"\nWrote policy to: {POLICY_OUT}")
