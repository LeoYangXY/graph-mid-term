#!/usr/bin/env python3
# Turn a raw temporal bipartite dataset (u, v, single timestamp) into the
# format the paper uses: each edge gets a duration drawn from a power law with
# alpha = -2.5, clipped to [1 h, 48 h].  Start = original timestamp,
# end = start + duration.
# Also emits a query file of random (s, t, window) pairs.
import sys, os, random, datetime

NL = chr(10)
H1 = 3600
H48 = 48 * 3600
ALPHA = -2.5

def plaw(rng):
    # inverse transform for p(x) ~ x^alpha, x in [xmin, inf), then clip
    xmin = H1
    u = rng.random()
    x = xmin * (1.0 - u) ** (1.0 / (ALPHA + 1.0))
    if x > H48:
        x = H48
    return int(x)

def parse_iso(s):
    try:
        return int(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp())
    except Exception:
        return None

def load(path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if not t or t[0] in "%#":
                continue
            parts = t.split("	") if "	" in t else t.split()
            if len(parts) < 3:
                continue
            if len(parts) >= 5:      # SNAP: user, isotime, lat, lon, loc
                ts = parse_iso(parts[1])
                if ts is None:
                    continue
                rows.append((int(parts[0]), int(parts[4]), ts))
            elif len(parts) >= 4:      # KONECT: user, page, weight, unixtime
                try:
                    rows.append((int(parts[0]), int(parts[1]), int(parts[3])))
                except Exception:
                    continue
    return rows

def main():
    src = sys.argv[1]
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 12345
    rng = random.Random(seed)
    rows = load(src)
    if not rows:
        print("no rows parsed from", src)
        return 1
    # remap ids to 0..n-1 separately for the two sides
    umap, vmap = {}, {}
    for u, v, t in rows:
        if u not in umap: umap[u] = len(umap)
        if v not in vmap: vmap[v] = len(vmap)
    nu, nv = len(umap), len(vmap)
    edges = [(umap[u], vmap[v], t) for u, v, t in rows]
    tmin = min(e[2] for e in edges)
    tmax = max(e[2] for e in edges)
    with open(out, "w") as f:
        f.write("# %d %d" % (nu, nv) + NL)
        for u, v, t in edges:
            f.write("%d %d %d %d" % (u, v, t, t + plaw(rng)) + NL)
    # queries: random pairs, window length = 50% of the time span (paper Exp-4)
    qf = out + ".query"
    span = tmax - tmin
    wl = int(span * 0.5)
    with open(qf, "w") as f:
        for _ in range(100000):
            s = rng.randrange(nu)
            d = rng.randrange(nu)
            a = rng.randrange(tmin, max(tmin + 1, tmax - wl))
            f.write("%d %d %d %d" % (s, d, a, a + wl) + NL)
    print("%s  |U|=%d |L|=%d |E|=%d  tmin=%d tmax=%d" % (os.path.basename(out), nu, nv, len(edges), tmin, tmax))
    return 0

sys.exit(main())
