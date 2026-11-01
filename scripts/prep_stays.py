#!/usr/bin/env python3
"""Turn message logs or Wi-Fi association logs into the stay table our builder eats.

Two sources are supported, both reproducing a published preprocessing so the
resulting graph can be compared with the original numbers:

  collegemsg  SNAP CollegeMsg.txt: "<sender> <receiver> <unix>".
              A message becomes a stay (sender, receiver, [t, t+3600]); a later
              message truncates the previous stay of the same sender and
              zero-length stays are dropped. Reproduces 1350 objects / 59661
              stays for the full set.

  wifi        Intelligent-WiFi association dump for one day, tab separated:
              asso_id user client_mac ap_id bytes RSSI conn disconn.
              Device -> object, AP -> location, [conn, disconn] -> stay. Consecutive
              sessions at the same AP are merged, records of one device at
              different APs are truncated, and the window is clipped. Reproduces
              10,003/2,332, 10,944/3,140 and 15,895/3,397 objects/locations.

Usage:
  prep_stays.py collegemsg CollegeMsg.txt out.txt
  prep_stays.py wifi 2019-05-06-on8.txt out.txt [HH:MM-HH:MM]
"""
import sys, calendar, datetime as dt
from collections import defaultdict

HOUR = 3600


def emit(path, stays, t0):
    objs = sorted({s[0] for s in stays})
    locs = sorted({s[1] for s in stays})
    om = {v: i for i, v in enumerate(objs)}
    lm = {v: i for i, v in enumerate(locs)}
    with open(path, 'w') as f:
        f.write('# %d %d\n' % (len(objs), len(locs)))
        for o, l, a, b in stays:
            f.write('%d %d %d %d\n' % (om[o], lm[l], a - t0, b - t0))
    print('%s: objects=%d locations=%d stays=%d' % (path, len(objs), len(locs), len(stays)))


def collegemsg(src, out):
    by = defaultdict(list)
    with open(src) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                by[int(p[0])].append((int(p[2]), int(p[1])))
    stays = []
    for sender, lst in by.items():
        lst.sort()
        for i, (t, r) in enumerate(lst):
            end = t + HOUR
            if i + 1 < len(lst) and lst[i + 1][0] < end:
                end = lst[i + 1][0]
            if end > t:
                stays.append((sender, r, t, end))
    emit(out, stays, min(s[2] for s in stays))


def wifi(src, out, window):
    def ts(s):
        return calendar.timegm(dt.datetime.strptime(s[:19], '%Y-%m-%dT%H:%M:%S').timetuple()) - 8 * HOUR

    a, b = window.split('-')
    day = '2019-05-06T'
    t1, t2 = ts(day + a + ':00'), ts(day + b + ':00')
    recs = []
    with open(src) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 8:
                continue
            s, e = ts(p[6]), ts(p[7])
            if s < t2 and e > t1:
                recs.append((p[2], p[3], max(s, t1), min(e, t2)))
    by = defaultdict(list)
    for dev, ap, s, e in recs:
        by[dev].append((s, e, ap))
    stays = []
    for dev, lst in by.items():
        lst.sort()
        merged = []
        for s, e, ap in lst:
            if merged and merged[-1][2] == ap and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e), ap)
            else:
                merged.append((s, e, ap))
        for i, (s, e, ap) in enumerate(merged):
            if i + 1 < len(merged) and merged[i + 1][0] < e:
                e = merged[i + 1][0]
            if e > s:
                stays.append((dev, ap, s, e))
    emit(out, stays, t1)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    if sys.argv[1] == 'collegemsg':
        collegemsg(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'wifi':
        wifi(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else '09:00-12:00')
    else:
        sys.exit(__doc__)
