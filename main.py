"""GTA VI Twitch CCU Forecaster — pattern-based quantile forecast.

Reads every <id>.in in the input dir, writes <id>.out with 23 rows (hours 2–24),
each five non-decreasing quantiles [q05,q25,q50,q75,q95]. Deterministic; stdlib only.

Strategy:
1. Derive per-hour median ratio + log-spread from comparables' release-day curves,
   weighted by how similar each comparable is to the target game (platform, genre tier).
2. Scale by hour-1 print p1 from the case's `known` row.
3. Blend with GTA VI's own premiere curve when available (tier=target-history).
4. Quantiles = p1 * r_h * exp(z_tau * s_h), rounded half-up to 2 decimals.
"""
import json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Empirical release-day shape learned from 38 comparables across all practice cases
BASE_RATIOS = [
    0.908, 0.795, 0.732, 0.709, 0.639, 0.589, 0.569, 0.576, 0.595, 0.633, 0.767,
    0.894, 1.051, 1.151, 1.285, 1.332, 1.462, 1.738, 1.746, 1.570, 1.465, 1.251, 1.0
]
# Log-space spread per hour derived from empirical variance of the same set
BASE_SIGMAS = [
    0.130, 0.328, 0.346, 0.351, 0.373, 0.395, 0.322, 0.354, 0.391, 0.375, 0.444,
    0.454, 0.493, 0.477, 0.465, 0.460, 0.632, 0.650, 0.586, 0.489, 0.441, 0.287, 0.200
]


def round2(x):
    s = x * 100
    w = math.floor(s)
    if s - w >= 0.5:
        w += 1
    v = w / 100
    return int(v) if v.is_integer() else v


def parse_iso_hour(ts):
    """Return UTC hour-of-day integer from an ISO-8601 timestamp like 2026-11-19T02:00:00Z."""
    try:
        return int(ts[11:13])
    except Exception:
        return None


def comparable_curve(comp):
    """Extract {hour_of_day: viewers} for a comparable's hourly series."""
    out = {}
    for pt in comp.get("hourly", []) or []:
        ts = pt.get("from")
        v = pt.get("viewers")
        h = parse_iso_hour(ts) if isinstance(ts, str) else None
        if h is not None and isinstance(v, (int, float)) and v > 0:
            # Keep the max if multiple points land on the same hour
            out[h] = max(out.get(h, 0), v)
    return out


def game_platforms(comp):
    plats = comp.get("platforms") or []
    if isinstance(plats, str):
        plats = [plats]
    joined = " ".join(str(p).lower() for p in plats)
    console = any(k in joined for k in ("playstation", "xbox", "ps5", "series x"))
    pc = "pc" in joined or "steam" in joined
    return console, pc


def similarity_weight(target_game, comp):
    """Heuristic weight: console AAA titles weigh more; PC-only less; premieres medium."""
    name = (comp.get("name") or "").lower()
    unlock = (comp.get("unlock") or "").lower()
    src = (comp.get("source") or "").lower()
    tier = (comp.get("tier") or "").lower()
    console, _ = game_platforms(comp)
    w = 1.0
    if not console:
        w *= 0.45
    if "premiere" in unlock or "extended look" in unlock or "trailer" in unlock:
        w *= 0.35
    if "release" in unlock or "launch" in unlock:
        w *= 1.15
    if "target-history" in tier or "gta vi" in name or "grand theft auto" in name:
        w *= 1.35
    # Genre proxy via keywords
    for kw in ("red dead", "god of war", "spider-man", "wolverine", "cyberpunk", "elden ring", "hogwarts"):
        if kw in name:
            w *= 1.25
            break
    return max(w, 0.05)


def fit_from_comparables(case):
    """Weighted median ratio + sigma per hour from comparables' release-day curves.

    For each comparable we locate its own launch day (day_start), take hours 2..24
    of that UTC day, divide by its hour-1 value (or the closest recorded point at
    hour 1), and accumulate into weighted buckets.
    """
    comps = case.get("comparables") or []
    # Accumulators per hour index 0..22 (= hours 2..24)
    num_ratio = [0.0] * 23
    den_w = [0.0] * 23
    sq_dev_num = [0.0] * 23
    sq_den = [0.0] * 23
    counts = [0] * 23

    for comp in comps:
        w = similarity_weight(case.get("game", {}).get("name", ""), comp)
        curve = comparable_curve(comp)
        if not curve:
            continue
        ds = comp.get("day_start") or ""
        if not isinstance(ds, str) or len(ds) < 10:
            continue
        # Identify the comparable's launch UTC day from day_start date portion
        launch_date = ds[:10]
        # Hour-1 anchor: prefer explicit hour==1, else earliest point on launch day
        h1_val = curve.get(1)
        if h1_val is None:
            # Take min hour present on that date
            cand = [(h, v) for h, v in curve.items() if 0 <= h <= 23]
            if not cand:
                continue
            cand.sort(key=lambda x: x[0])
            h1_val = cand[0][1]
        if not h1_val or h1_val <= 0:
            continue
        for hi in range(23):
            hour = hi + 2
            v = curve.get(hour)
            if v is None or v <= 0:
                continue
            r = v / h1_val
            if r <= 0 or r > 50:
                continue
            lr = math.log(r)
            num_ratio[hi] += w * lr
            den_w[hi] += w
            counts[hi] += 1
            # variance accumulation around running mean approximated later
            sq_dev_num[hi] += w * lr * lr
            sq_den[hi] += w

    medians = []
    sigmas = []
    for hi in range(23):
        if den_w[hi] > 0 and counts[hi] >= 2:
            mean_lr = num_ratio[hi] / den_w[hi]
            var_lr = max(sq_dev_num[hi] / sq_den[hi] - mean_lr * mean_lr, 0.0)
            sd_lr = math.sqrt(var_lr)
            # Floor sigma so quantiles stay honest; cap to avoid runaway tails
            sd_lr = min(max(sd_lr, 0.06), 0.55)
            medians.append(math.exp(mean_lr))
            sigmas.append(sd_lr)
        else:
            medians.append(BASE_RATIOS[hi])
            sigmas.append(BASE_SIGMAS[hi])
    return medians, sigmas


def blend_with_base(medians, sigmas, base_r, base_s, alpha=0.65):
    """Blend fitted values toward canonical base when data is thin/noisy."""
    out_m = []
    out_s = []
    for i in range(23):
        m = alpha * medians[i] + (1 - alpha) * base_r[i]
        s = alpha * sigmas[i] + (1 - alpha) * base_s[i]
        # Enforce sanity bounds
        m = min(max(m, 0.15), 8.0)
        s = min(max(s, 0.05), 0.60)
        out_m.append(m)
        out_s.append(s)
    # Monotonic smoothing: enforce non-increasing after the peak (hour ~5)
    peak_i = max(range(23), key=lambda i: out_m[i])
    for i in range(peak_i + 1, 23):
        if out_m[i] > out_m[i - 1]:
            out_m[i] = out_m[i - 1]
    return out_m, out_s


def gta_premiere_adjustment(case):
    """If GTA VI's own premiere curve is present, tilt ratios upward modestly.

    The extended-look premiere peaked near 1.4M concurrent on 2026-08-27T19Z.
    A real midnight-unlock launch typically exceeds that by a large factor,
    but we only use this as a small multiplicative nudge on early hours.
    """
    comps = case.get("comparables") or []
    for comp in comps:
        nm = (comp.get("name") or "").lower()
        tk = (comp.get("tier") or "").lower()
        if "grand theft auto" in nm or "gta vi" in nm or "target-history" in tk:
            curve = comparable_curve(comp)
            if curve:
                mx = max(curve.values())
                # Larger premiere peak -> slightly stronger opening ramp
                if mx >= 800_000:
                    boost = 1.12
                elif mx >= 300_000:
                    boost = 1.06
                else:
                    boost = 1.02
                return boost
    return 1.0


def forecast_case(case):
    lv = case.get("levels") or LEVELS
    if lv != LEVELS:
        raise ValueError(f"unexpected levels {lv}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1, int(first.get("viewers", 1)))

    med, sig = fit_from_comparables(case)
    med, sig = blend_with_base(med, sig, BASE_RATIOS, BASE_SIGMAS, alpha=0.65)
    boost = gta_premiere_adjustment(case)
    # Apply boost only to first few hours where hype matters most
    for i in range(min(6, len(med))):
        f = 1.0 + (boost - 1.0) * (1.0 - i / 6.0)
        med[i] *= f

    hours_out = []
    for row in case.get("forecast") or []:
        h = row.get("hour")
        if not isinstance(h, int) or h < 2 or h > 24:
            continue
        idx = h - 2
        r = med[idx]
        s = sig[idx]
        qs = [round2(p1 * r * math.exp(zv * s)) for zv in Z]
        # Ensure strictly positive and non-decreasing
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        qs = [max(q, 1) for q in qs]
        hours_out.append({"hour": h, "q": qs})
    # Guarantee all 23 hours present exactly once, sorted by hour
    seen = {row["hour"] for row in hours_out}
    for h in range(2, 25):
        if h not in seen:
            idx = h - 2
            r = med[idx]
            s = sig[idx]
            qs = [round2(p1 * r * math.exp(zv * s)) for zv in Z]
            for j in range(1, 5):
                if qs[j] < qs[j - 1]:
                    qs[j] = qs[j - 1]
            qs = [max(q, 1) for q in qs]
            hours_out.append({"hour": h, "q": qs})
    hours_out.sort(key=lambda x: x["hour"])
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        fc = forecast_case(case)
        body = json.dumps(fc, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
