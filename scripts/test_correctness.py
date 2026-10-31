#!/usr/bin/env python3
"""Independent exhaustive oracle for tiny graphs; no production BFS is reused."""
import argparse
import itertools
from pathlib import Path
import random
import struct
import subprocess
import tempfile


def run(args, ok=True):
    r = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if (r.returncode == 0) != ok:
        raise AssertionError((args, r.returncode, r.stdout, r.stderr))
    return r.stdout + r.stderr


def write_graph(path, n, edges):
    with path.open('wb') as f:
        f.write(struct.pack('<IQ', n, len(edges)))
        for c in range(4):
            f.write(struct.pack('<' + 'I' * len(edges), *(e[c] for e in edges)))


def read_index(path):
    data = path.read_bytes()
    marker, version, n, k = struct.unpack_from('<4I', data)
    assert marker == 0xffffffff and version == 1
    pos = 16
    lin, lout = [], []
    for _ in range(n):
        for lists in (lin, lout):
            count, = struct.unpack_from('<I', data, pos)
            pos += 4
            labels = [struct.unpack_from('<4I', data, pos + 16*i) for i in range(count)]
            pos += 16*count
            lists.append(labels)
    assert pos == len(data)
    return lin, lout, k


def paths(n, edges, k):
    # Enumerate actual walks, retaining every distinct (destination,start,end).
    adj = [[] for _ in range(n)]
    for u, v, a, b in edges:
        adj[u].append((v, a, b)); adj[v].append((u, a, b))
    all_paths = []
    for s in range(n):
        found = set()
        layer = set(adj[s])
        for h in range(1, k+1):
            found.update((v, a, b, h) for v, a, b in layer)
            layer = {(y, a, d) for v, a, b in layer for y, c, d in adj[v] if c >= b}
        all_paths.append(found)
    return all_paths


def indexed(lin, lout, s, t, start, end, k):
    if s == t:
        return True
    for w, a, b, h in lout[s]:
        if w == t and start <= a and b <= end and h <= k:
            return True
    for w, a, b, h in lin[t]:
        if w == s and start <= a and b <= end and h <= k:
            return True
    return any(w == z and start <= a and b <= c and d <= end and h+j <= k
               for w, a, b, h in lout[s] for z, c, d, j in lin[t])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--builder', required=True, type=Path)
    p.add_argument('--reader', required=True, type=Path)
    args = p.parse_args()
    builder, reader = args.builder.resolve(), args.reader.resolve()
    rng = random.Random(73019)
    cases = [('empty', 3, [], 3), ('singleton', 1, [], 1),
             ('chain', 6, [(i, i+1, i, i+1) for i in range(5)], 5),
             ('tradeoff', 5, [(0,1,0,2),(0,1,4,6),(1,2,2,3),(1,2,6,7),
                              (2,3,3,4),(2,4,7,8),(0,4,1,7),(4,3,8,9)], 5),
             ('post2038', 3, [(0,1,3000000000,3000000001),
                              (1,2,3000000001,3000000002)], 3)]
    for i in range(10):
        n = rng.randrange(2, 8)
        edges = []
        for _ in range(rng.randrange(4, 20)):
            u, v = rng.sample(range(n), 2)
            a = rng.randrange(0, 10)
            edges.append((u, v, a, a+rng.randrange(1, 4)))
        cases.append((f'random{i}', n, edges, 5))
    checked = 0
    with tempfile.TemporaryDirectory(prefix='correctness-') as tmp:
        root = Path(tmp)
        for name, n, edges, K in cases:
            graph = root/'graph.bin'; index = root/'index.bin'; queries = root/'queries.txt'
            write_graph(graph, n, edges)
            oracle = paths(n, edges, K)
            times = sorted({0, *(t for e in edges for t in e[2:])})
            cases_q = [(s,t,a,b,k) for s,t in itertools.product(range(n), repeat=2)
                       for a,b in itertools.combinations_with_replacement(times, 2) for k in range(K+1)]
            expected = [s==t or any(v==t and a<=x and y<=b and h<=k for v,x,y,h in oracle[s])
                        for s,t,a,b,k in cases_q]
            for batch in (1, 3, 8):
                run([builder, graph, batch, K, index])
                lin, lout, horizon = read_index(index)
                assert horizon == K
                for v in range(n):
                    assert all((v,a,b,h) in oracle[w] for w,a,b,h in lin[v])
                    assert all((w,a,b,h) in oracle[v] for w,a,b,h in lout[v])
                for q, want in zip(cases_q, expected):
                    assert indexed(lin,lout,*q) == want, (name,batch,q,want)
                checked += len(cases_q)
            # Exercise the C++ label reader and production online oracle too.
            selected = [q for q in cases_q if q[-1] == K]
            queries.write_text(''.join(f'{s} {t} {a} {b}\n' for s,t,a,b,_ in selected))
            run([reader, 'cmp', index, graph, queries, K])
            run([reader, 'iquery', index, queries, K+1], ok=False)
            print(f'{name}: labels sound; all pairs/windows/k agree; batches 1/3/8')
        # A 2D Pareto frontier can have arbitrarily many incomparable intervals.
        edges = [(0,1,3*i,3*i+1) for i in range(300)]
        write_graph(graph, 2, edges)
        log = run([builder, graph, 2, 1, index])
        assert 'retry' in log
        lin,lout,_ = read_index(index)
        assert all(indexed(lin,lout,0,1,a,b,1) for _,_,a,b in edges)
        queries.write_text(''.join(f'0 1 {a} {b}\n' for _,_,a,b in edges))
        run([reader,'cmp',index,graph,queries,1])
        # Invalid inputs must fail, not emit successful-but-incomplete results.
        old_index = index.read_bytes()
        run([builder,graph,0,1,index],ok=False)
        run([builder,graph,2,999,index],ok=False)
        assert index.read_bytes() == old_index
        graph.write_bytes(graph.read_bytes()[:-1])
        run([builder,graph,2,1,index],ok=False)
        assert index.read_bytes() == old_index
        index.write_bytes(old_index[:-1])
        run([reader,'iquery',index,queries,1],ok=False)
    print(f'PASS: {checked} exhaustive query comparisons + 300-interval overflow + invalid input/horizon tests')


if __name__ == '__main__':
    main()
