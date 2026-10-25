import numpy as np, pandas as pd
from itertools import product
from sklearn.metrics import f1_score

OUT = "out"
y = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]).AGE_RANGE.values
names = ["lgb_fe", "lr", "rf"]
oof = {n: np.load(f"{OUT}/oof_{n}.npy") for n in names}
tst = {n: np.load(f"{OUT}/test_{n}.npy") for n in names}
mf1 = lambda p, yy=y: f1_score(yy, p.argmax(1), average="macro")

for n in names: print(n, round(mf1(oof[n]), 4))

# pesos do blend (grelha simples)
best = (0, None)
for w in product(np.arange(0, 1.01, 0.1), repeat=2):
    if sum(w) > 1: continue
    ww = (w[0], w[1], 1 - sum(w))
    p = sum(a * oof[n] for a, n in zip(ww, names))
    s = mf1(p)
    if s > best[0]: best = (s, ww)
print("melhor blend", best)
ww = best[1]
P_oof = sum(a * oof[n] for a, n in zip(ww, names))
P_te = sum(a * tst[n] for a, n in zip(ww, names))

# multiplicadores por classe para Macro F1 (coordinate ascent), validados em 2 metades
def fit_mult(P, yy, iters=3, grid=np.linspace(0.5, 2.0, 16)):
    m = np.ones(P.shape[1])
    for _ in range(iters):
        for k in range(P.shape[1]):
            sc = []
            for g in grid:
                mm = m.copy(); mm[k] = g
                sc.append(mf1(P * mm, yy))
            m[k] = grid[int(np.argmax(sc))]
    return m

rng = np.random.RandomState(0); idx = rng.permutation(len(y)); a, b = idx[: len(y)//2], idx[len(y)//2:]
ma = fit_mult(P_oof[a], y[a]); mb = fit_mult(P_oof[b], y[b])
print("sem mult  (metade b):", round(mf1(P_oof[b], y[b]), 4), "| com mult de a:", round(mf1(P_oof[b] * ma, y[b]), 4))
print("sem mult  (metade a):", round(mf1(P_oof[a], y[a]), 4), "| com mult de b:", round(mf1(P_oof[a] * mb, y[a]), 4))

m_all = fit_mult(P_oof, y)
print("OOF blend:", round(mf1(P_oof), 4), "| com mult (otimista):", round(mf1(P_oof * m_all), 4))

pred = (P_te * m_all).argmax(1)
sub = pd.read_csv("example_submission_10.csv")
assert len(sub) == len(pred) == 3655
sub["AGE_RANGE"] = pred
sub.to_csv(f"{OUT}/submission_v1.csv", index=False)
print(sub.AGE_RANGE.value_counts().sort_index().to_dict())
np.save(f"{OUT}/oof_blend.npy", P_oof); np.save(f"{OUT}/class_mult.npy", m_all)
