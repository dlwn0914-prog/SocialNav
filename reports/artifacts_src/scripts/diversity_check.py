"""Candidate diversity: Crowd-FM (v3 refined / raw) vs codebook, same measures on a common 2.4 s horizon."""
import glob, json, sys
import numpy as np, torch
sys.path.insert(0, "/home/nuri2/SocialNav")
from src.preference.traj_vqvae import TrajVQVAE
from utils.train_code_prior import load_prior, situation

T = np.array([0.8, 1.6, 2.4])

def measures(paths, ppl):
    """paths (K,3,2) at T; ppl (M,4) x,y,vx,vy. Reference person = closest to the mean path."""
    out = {}
    end = paths[:, -1]
    out["lat_spread"] = end[:, 1].std()
    out["end_spread"] = np.sqrt(((end - end.mean(0)) ** 2).sum(1)).mean()
    pair = np.linalg.norm(paths[:, None] - paths[None], axis=-1).mean(-1)
    out["pair_dist"] = pair[np.triu_indices(len(paths), 1)].mean()
    if len(ppl) == 0:
        out.update(side_both=np.nan, front_both=np.nan, n_safe=len(paths), near=np.nan); return out
    traj = ppl[:, None, :2] + T[None, :, None] * ppl[:, None, 2:]          # (M,3,2)
    dmean = np.linalg.norm(paths.mean(0)[None] - traj, axis=-1).min(1)
    j = dmean.argmin()
    gap = paths - traj[j][None]                                           # (K,3,2)
    k = np.linalg.norm(gap, axis=-1).argmin(1)
    g = gap[np.arange(len(paths)), k]
    sp = np.linalg.norm(ppl[j, 2:]); vh = ppl[j, 2:] / max(sp, 1e-6)
    side = np.tanh(g[:, 1] / 0.5); front = np.tanh(g @ vh / 0.5) if sp > 0.2 else np.zeros(len(paths))
    dmin = np.linalg.norm(paths[:, None] - traj[None], axis=-1).min((1, 2))
    safe = dmin >= 0.5
    out["near"] = dmean[j] < 1.0
    out["n_safe"] = safe.sum()
    s, f = side[safe], front[safe]
    out["side_both"] = float(len(s) > 1 and s.max() > 0.3 and s.min() < -0.3)
    out["front_both"] = float(len(f) > 1 and f.max() > 0.3 and f.min() < -0.3)
    out["side_both_all"] = float(side.max() > 0.3 and side.min() < -0.3)
    return out

def interp(c, dur=3.0):
    t = np.linspace(0, dur, c.shape[1])
    return np.stack([np.stack([np.interp(T, t, ci[:, a]) for a in (0, 1)], -1) for ci in c])

rows = {}
for name, pat, key in [("crowdfm_v3_refined", "/home/nuri2/crowd_fm_ws/scenarios_v3/gc_all/*.json", "candidates"),
                       ("crowdfm_v3_raw", "/home/nuri2/crowd_fm_ws/scenarios_v3/gc_all/*.json", "raw_candidates"),
                       ("crowdfm_gcp_refined", "/home/nuri2/crowd_fm_ws/scenarios_gcp/v3/*.json", "candidates")]:
    R = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        if key not in d: continue
        c = np.array(d[key]); p = np.array(d["people"]).reshape(-1, 2); v = np.array(d.get("people_vel", np.zeros_like(p))).reshape(-1, 2)
        R.append(measures(interp(c, d.get("traj_duration", 3.0)), np.c_[p, v]))
    rows[name] = R

# codebook: GC test, far destination, 2000 random situations, 16 code paths cut to 2.4 s
net, pa = load_prior("/home/nuri2/SocialNav/results/codebook/goalfix/gc_dest_s1.pt")
ck = torch.load("/home/nuri2/SocialNav/" + pa["ckpt"] if not pa["ckpt"].startswith("/") else pa["ckpt"], weights_only=False)
d = dict(np.load(pa["data"]))
vq = TrajVQVAE(horizon=d["future"].shape[1], num_codes=ck["args"]["codes"], latent=ck["args"]["latent"]); vq.load_state_dict(ck["state_dict"]); vq.eval()
with torch.no_grad(): dev = vq.decode_codes().numpy()[:, :3]
_, x = situation(d, "heading", True, pa["people_input"], pa["goal"])
base = np.where((d["split"] == 2) & (d["dest_t"] >= pa["min_dest_t"]))[0]
rng = np.random.default_rng(0)
for name, sel in [("codebook_all", base), ("codebook_conflict", base[x[base][:, -4] < 8])]:
    R = []
    for s in rng.choice(sel, 2000, replace=False):
        paths = np.stack([d["speed"][s] * T, np.zeros(3)], -1)[None] + dev
        R.append(measures(paths, d["people"][s][: d["n_people"][s]]))
    rows[name] = R

print(f"{'':22s} {'n':>5s} {'lat_sd':>7s} {'end_sp':>7s} {'pair':>6s} {'near':>5s} {'safe':>5s} {'side2':>6s} {'side2*':>6s} {'front2':>6s}")
for name, R in rows.items():
    m = lambda k: np.nanmean([r.get(k, np.nan) for r in R])
    print(f"{name:22s} {len(R):5d} {m('lat_spread'):7.3f} {m('end_spread'):7.3f} {m('pair_dist'):6.3f} {m('near'):5.2f} {m('n_safe'):5.1f} {m('side_both'):6.2f} {m('side_both_all'):6.2f} {m('front_both'):6.2f}")
    if name.startswith("crowdfm"):
        nr = [r for r in R if r.get("near") == True]
        if nr: print(f"{'  near-person only':22s} {len(nr):5d} {np.mean([r['lat_spread'] for r in nr]):7.3f} {'':7s} {'':6s} {'':5s} {np.mean([r['n_safe'] for r in nr]):5.1f} {np.mean([r['side_both'] for r in nr]):6.2f} {np.mean([r['side_both_all'] for r in nr]):6.2f} {np.mean([r['front_both'] for r in nr]):6.2f}")
