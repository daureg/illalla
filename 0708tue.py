# coding: utf-8
import approx_emd as ap
import datetime as dt
import numpy as np
import re
import itertools
with open('runningtime.log') as infile:
    running = infile.readlines()
CITIES = sorted(['barcelona', 'sanfrancisco', 'rome', 'newyork', 'washington', 'berlin', 'paris'])
NEIGHBORHOODS = sorted(['triangle', 'latin', 'montmartre', 'pigalle', 'marais', 'official', 'weekend', '16th'])
METRICS = sorted(['jsd', 'emd', 'cluster', 'emd-lmnn', 'leftover'])
RADIUS = map(str, np.linspace(200, 500, 5).astype(int))
FILENAME = re.compile(r'comparaison_([a-z]+)/([a-z]+)_([a-z16]+)_(\d+)_([a-z-]+).my')
res = {q: {t: {n: {m: {r: None for r in RADIUS}
                   for m in METRICS }
               for n in NEIGHBORHOODS }
           for t in CITIES if t != q }
       for q in CITIES if q != 'berlin' }
def parse_line(line):
    starting = 'will write' in line
    triton = 'slurm' in line
    when = dt.datetime.strptime(line.split('[')[0].strip(), '%Y-%m-%d %H:%M:%S,%f')
    infos = FILENAME.search(line).groups()
    return [starting, when, triton] + list(infos)
start_time = None
for line in running:
    infos = parse_line(line)
    starting, when, triton, query_city, target_city, district, radius, metric = infos
    if starting:
        start_time = when
    else:
        res[query_city][target_city][district][metric][radius] = (when - start_time).total_seconds()
def get_slice(tuple_request):
    q, t, d, m, r = tuple_request
    qq = [_ for _ in CITIES if _ != 'berlin'] if not q else [q]
    tt = CITIES if not t else [t]
    dd = NEIGHBORHOODS if not d else [d]
    mm = METRICS if not m else [m]
    rr = RADIUS if not r else [r]
    prod = itertools.product(qq, tt, dd, mm)
    vals = []
    queries = []
    for qq, tt, dd, mm in prod:
        if qq == tt:
            continue
        this_vals = []
        for rr in RADIUS:
            duration = res[qq][tt][dd][mm][rr]
            if duration:
                this_vals.append(duration)
        if this_vals:
            queries.append((qq, tt, dd))
            vals.append(sum(this_vals))
    return vals, queries
emd, AQ = get_slice((None, None, None, 'emd', None))
emd = np.array(emd)
queries = lambda src: [_[1:] for _ in AQ if _[0] == src]
from operator import itemgetter
QCITIES = ['barcelona', 'newyork', 'paris', 'rome', 'sanfrancisco', 'washington']
full_data = []
# for n_step in range(5):
for n_step in [1]:
    Adsts, Atms, Arr = [], [], []
    t_fast, t_slow, fast, slow = np.array([]), np.array([]), np.array([]), np.array([])
    for qcity in QCITIES:
        ALL_Q = queries(qcity)
        dsts, tms, rrs = ap.test_all_queries(ALL_Q, qcity, n_steps=n_step,
                                             k=50)
        Adsts.append(dsts)
        Atms.append(tms)
        Arr.append(rrs)
        with open('static/cmp_{}.json'.format(qcity)) as infile:
            star = ap.ujson.load(infile)
        get_gold = lambda c, d: [_['dst'] for _ in star[c][d] if _['metric'] == 'emd']
        rq = [star.get(_[0], {}).get(_[1]) is not None and np.min(get_gold(*_)) < 1e5
              and len(dsts[i]) > 0
              for i, _ in enumerate(ALL_Q)]
        rqs = list(itertools.compress(ALL_Q, rq))
        t_slow = np.hstack([t_slow, np.array([t for t, q in zip(emd, AQ) if q[0] == qcity and q[1:] in rqs])])
        t_fast = np.hstack([t_fast, np.array(list(itertools.compress(tms, rq)))])
        slow = np.hstack([slow, np.array([np.min(get_gold(*q)) for q in itertools.compress(ALL_Q, rq)])])
        fast = np.hstack([fast, np.array([10 if len(_) == 0 else min(_) for _ in itertools.compress(dsts, rq)])])
    full_data.append((Adsts, Atms, Arr, t_fast, t_slow, fast, slow))
import persistent as p
p.save_var('approx_brute_relevance.my', full_data)
import sys
sys.exit()
del full_data[:]
full_data = []
n_step = 1
# for n_step in range(5):
for knn in [8, 25, 50, 80, 160]:
    Adsts, Atms, Arr = [], [], []
    t_fast, t_slow, fast, slow = np.array([]), np.array([]), np.array([]), np.array([])
    for qcity in QCITIES:
        ALL_Q = queries(qcity)
        dsts, tms, rrs = ap.test_all_queries(ALL_Q, qcity, n_steps=1, k=knn)
        Adsts.append(dsts)
        Atms.append(tms)
        with open('static/cmp_{}.json'.format(qcity)) as infile:
            star = ap.ujson.load(infile)
        get_gold = lambda c, d: [_['dst'] for _ in star[c][d] if _['metric'] == 'emd']
        rq = [star.get(_[0], {}).get(_[1]) is not None and np.min(get_gold(*_)) < 1e5
              and len(dsts[i]) > 0
              for i, _ in enumerate(ALL_Q)]
        rqs = list(itertools.compress(ALL_Q, rq))
        t_slow = np.hstack([t_slow, np.array([t for t, q in zip(emd, AQ) if q[0] == qcity and q[1:] in rqs])])
        t_fast = np.hstack([t_fast, np.array(list(itertools.compress(tms, rq)))])
        slow = np.hstack([slow, np.array([np.min(get_gold(*q)) for q in itertools.compress(ALL_Q, rq)])])
        fast = np.hstack([fast, np.array([10 if len(_) == 0 else min(_) for _ in itertools.compress(dsts, rq)])])
    full_data.append((Adsts, Atms, Arr, t_fast, t_slow, fast, slow))
p.save_var('new_varying_k.my', full_data)
# p.save_var('five_larger_steps.my', full_data)
# ratio = np.sort(fast/slow)[:-2]
# ppl.boxplot(np.reshape(ratio, (ratio.size, 1)), xticklabels=['default param'], notch=True, bootstrap=None)
# plt.hlines(1.0, 0, len(ratio), color=ppl.colors.almost_black)
# plt.figure(figsize=(30, 18))
# ratios = np.zeros((len(full_data[0][-1])-2, 5))
# for nstep in range(5):
#     fast, slow = full_data[nstep][-2], full_data[nstep][-1]
#     ratio = np.sort(fast/slow)[:-2]
#     ratios[:, nstep] = ratio
# ppl.boxplot(ratios,  notch=True, bootstrap=1000)
# plt.hlines(1.0, 0, 5.3, color=ppl.colors.almost_black)
# plt.figure(figsize=(30, 18))
# tratios = np.zeros((len(full_data[0][-1])-2, 5))
# for nstep in range(5):
#     tfast, tslow = full_data[nstep][-4], full_data[nstep][-3]
#     ratio = np.sort(tslow/tfast)[:-2]
#     tratios[:, nstep] = ratio
#     #ratios.append(np.reshape(ratio, (ratio.size, 1)))
# ppl.boxplot(tratios,  notch=True, bootstrap=1000)
# #plt.hlines(1.0, 0, 5.3, color=ppl.colors.almost_black)
