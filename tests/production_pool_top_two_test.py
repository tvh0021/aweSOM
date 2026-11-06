"""Top-two sheet selection (user-approved 2026-11-06) in production_pool.select_group."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('production_pool_top_two', ROOT/'examples/grmhd-bbh/production_pool.py')
pool = importlib.util.module_from_spec(spec); spec.loader.exec_module(pool)

# The 1e9 production collect (job 8583, 91x64): seed_overlap["42-43"]["jaccard"] from
# freeze-manifest.json and E(log_chi_B) of the candidates from 91x64-s42.json / 91x64-s43.json.
J1E9 = np.array([
    [0.0, 0.07235673290206353, 0.06450602118859096, 0.09480026302784579, 0.042494135918048945, 5.216332797024771e-09, 0.008854749117493026, 0.03363258496406233, 0.014028179562842465, 0.25837262506784714],
    [0.0, 0.0, 0.013500302239459114, 3.934336877492843e-09, 0.149395346014643, 2.8880509850744802e-08, 0.0, 0.0, 0.0, 0.003084488646510203],
    [0.0, 0.11014289020169456, 0.0, 0.013650902237574912, 0.002244013376946831, 0.0, 0.0, 0.33199683309408146, 0.04172207254875248, 0.0010709970520572064],
    [0.0028576012241974522, 0.010942338407811265, 0.08513804110282766, 0.371043233128592, 0.2563438994083049, 0.00708875222112384, 0.0008843562749010974, 4.513812474270035e-06, 0.0036223549993959574, 0.027392766443888574],
    [9.135833889637098e-09, 6.168549300903587e-07, 0.010100999484959062, 0.0346863958159691, 0.04562338017912095, 0.0, 0.08787392568066872, 0.0003212697852451972, 0.0, 0.10843832342434483],
    [0.0, 0.0, 0.05890873581532604, 0.000532654424366365, 0.004123159421599213, 0.0, 0.03088087738530019, 0.0, 0.060820867562916506, 0.051678827567604245],
    [0.0, 0.009897920773826946, 1.8370931473049945e-08, 0.2860532558723188, 2.5488218778097873e-07, 1.0233619094525084e-08, 0.0, 0.002107802448097683, 0.08444708872359595, 0.0014814824295032013],
    [0.0, 4.1233909396356846e-05, 2.8531755139439874e-06, 0.0, 0.0488828306600336, 4.09127581799605e-06, 0.04447324392203701, 0.01015685024699114, 0.03283805265897358, 0.07272757636608636],
    [0.0010529608050206052, 0.0, 0.0067258250882253085, 0.000390871778124725, 1.7140323435760693e-08, 0.8605843733607133, 0.0, 0.0, 1.7099639617322502e-05, 1.1290620314412294e-05],
    [0.88196134973516, 4.4279191059473406e-08, 0.0, 0.010171739864886719, 4.08640428288561e-05, 0.009833373975841221, 0.0, 0.0, 1.1053314801787869e-08, 9.72901248012614e-07],
])
E1E9 = {42: {0: 0.47020974696909934, 1: 0.22964808416884466, 2: 0.8017402180761716, 4: 0.255794404395674, 5: 0.6034769230707385, 6: 0.11306233108592399, 7: 0.8743154190853559},
        43: {1: 0.6439273765655079, 6: 0.877343099969367, 7: 0.8931579525007419, 8: 0.7270590916833601, 9: 0.5218803662586152}}
ORDERS1E9 = {42: [7, 2, 5, 0, 4, 1, 6], 43: [7, 6, 8, 1, 9]}


def labelling(*blocks):
    return np.concatenate([np.full(n, c) for c, n in blocks])


def jaccards(labels):
    k = {s: int(l.max())+1 for s, l in labels.items()}
    return {(i, j): pool.jaccard_matrix(labels[i], labels[j], k[i], k[j], chunk=7)
            for i, j in itertools.combinations(labels, 2)}


def test_1e9_production_case():
    orders = {s: sorted(E1E9[s], key=lambda c: (-E1E9[s][c], c)) for s in E1E9}
    assert orders == ORDERS1E9
    picks, audit = pool.select_group(ORDERS1E9, {(42, 43): J1E9}, E1E9)
    assert audit['status'] == 'selected' and picks == {42: [2], 43: [7]}
    assert audit['first']['candidates'] == {42: 2, 43: 7}
    assert audit['first']['min_E_log_chi_B'] == E1E9[42][2]
    assert audit['first']['pairs']['42-43']['jaccard'] == pytest.approx(0.33199683309408146, rel=0, abs=0)
    # the four tuples in candidate order; only 2/7 is mutual
    assert [t['candidates'] for t in audit['tuples']] == [{42: 7, 43: 7}, {42: 7, 43: 6}, {42: 2, 43: 7}, {42: 2, 43: 6}]
    assert [t['all_mutual'] for t in audit['tuples']] == [False, False, True, False]
    assert audit['eligible'] == {42: [dict(cluster=7, E_log_chi_B=E1E9[42][7]), dict(cluster=2, E_log_chi_B=E1E9[42][2])],
                                 43: [dict(cluster=7, E_log_chi_B=E1E9[43][7]), dict(cluster=6, E_log_chi_B=E1E9[43][6])]}
    # no second pick: the unused 42:7 / 43:6 are not mutual
    assert audit['second']['candidates'] == {42: 7, 43: 6} and not audit['second']['reproduced']
    assert 'not mutual' in audit['reason']
    prev = audit['approved_rule_outcome']
    assert prev['status'] == 'none' and prev['selected'] == {42: [], 43: []} and 'not mutual' in prev['reason']
    assert prev['rule'] == pool.PREVIOUS_SELECTION_RULE and audit['rule'] == pool.SELECTION_RULE


def test_exact_tie_keeps_first_enumerated():
    # 42:0 <-> 43:1 and 42:1 <-> 43:0 are mutual; both tuples score min E = 0.5
    labels = {42: labelling((0, 40), (1, 30), (2, 30)), 43: labelling((1, 40), (0, 30), (2, 30))}
    J = jaccards(labels)
    orders = {42: [0, 1], 43: [0, 1]}
    chi = {42: {0: .9, 1: .5}, 43: {0: .8, 1: .5}}
    picks, audit = pool.select_group(orders, J, chi)
    assert [t['min_E_log_chi_B'] for t in audit['tuples'] if t['all_mutual']] == [.5, .5]
    assert audit['first']['candidates'] == {42: 0, 43: 1}
    # the residual pair is mutual, so it is the second pick
    assert picks == {42: [0, 1], 43: [1, 0]} and audit['second']['reproduced']
    assert audit['approved_rule_outcome']['status'] == 'none'
    # break the tie: the larger minimum wins regardless of enumeration order
    chi = {42: {0: .9, 1: .5}, 43: {0: .8, 1: .4}}
    picks, audit = pool.select_group(orders, J, chi)
    assert audit['first']['candidates'] == {42: 1, 43: 0} and picks == {42: [1, 0], 43: [0, 1]}


# Three seeds on 100 rows: 42:0<->43:2<->44:1, 42:1<->43:0<->44:0, 42:2<->43:3<->44:2, 42:3<->43:1<->44:3.
SEEDS = {42: labelling((0, 40), (1, 30), (2, 20), (3, 10)),
         43: labelling((2, 40), (0, 30), (3, 20), (1, 10)),
         44: labelling((1, 38), (0, 32), (2, 20), (3, 10))}


def test_three_seeds_pairwise_and_second_absent():
    orders = {42: [1, 0], 43: [0, 2], 44: [2, 0]}
    chi = {42: {1: .9, 0: .8}, 43: {0: .9, 2: .8}, 44: {2: .9, 0: .8}}
    picks, audit = pool.select_group(orders, jaccards(SEEDS), chi)
    # 1/0/2 is mutual for 42-43 only: rejected although the first pair agrees
    t = audit['tuples'][0]
    assert t['candidates'] == {42: 1, 43: 0, 44: 2} and not t['all_mutual']
    assert t['mutual'] == {'42-43': True, '42-44': False, '43-44': False}
    assert [x['candidates'] for x in audit['tuples'] if x['all_mutual']] == [{42: 1, 43: 0, 44: 0}]
    assert picks == {42: [1], 43: [0], 44: [0]} and audit['first']['min_E_log_chi_B'] == .8
    # residual 0/2/2: 42-43 mutual, the pairs with 44 are not, so no second pick
    assert audit['second']['candidates'] == {42: 0, 43: 2, 44: 2} and not audit['second']['reproduced']
    assert {k for k, p in audit['second']['pairs'].items() if not p['mutual']} == {'42-44', '43-44'}
    assert audit['approved_rule_outcome']['status'] == 'none'


def test_three_seeds_second_present_after_mixed_first():
    orders = {42: [3, 1], 43: [0, 1], 44: [0, 3]}
    chi = {42: {3: .9, 1: .7}, 43: {0: .8, 1: .6}, 44: {0: .85, 3: .5}}
    picks, audit = pool.select_group(orders, jaccards(SEEDS), chi)
    # mutual tuples: 3/1/3 (min 0.5) and 1/0/0 (min 0.7); the larger minimum wins
    assert [(x['candidates'], x['min_E_log_chi_B']) for x in audit['tuples'] if x['all_mutual']] == [
        ({42: 3, 43: 1, 44: 3}, .5), ({42: 1, 43: 0, 44: 0}, .7)]
    assert picks == {42: [1, 3], 43: [0, 1], 44: [0, 3]} and audit['second']['reproduced']
    assert audit['status'] == 'selected' and audit['approved_rule_outcome']['status'] == 'none'


def test_one_eligible_seed_has_no_second():
    orders = {42: [1, 3], 43: [0], 44: [0, 3]}
    chi = {42: {1: .9, 3: .8}, 43: {0: .9}, 44: {0: .9, 3: .8}}
    picks, audit = pool.select_group(orders, jaccards(SEEDS), chi)
    assert picks == {42: [1], 43: [0], 44: [0]} and audit['second']['missing_in_seeds'] == [43]
    assert len(audit['tuples']) == 4


@pytest.mark.parametrize('trial', range(200))
def test_previous_rule_success_is_preserved(trial):
    """Whenever the previous rule selects, the top-two rule gives the same picks, first and second."""
    rng = np.random.default_rng(trial)
    n_seeds = int(rng.integers(2, 4)); k = int(rng.integers(3, 7)); n = 300
    base = rng.integers(0, k, n)
    labels = {}
    for s in range(42, 42+n_seeds):
        noisy = np.where(rng.random(n) < rng.uniform(0, .5), rng.integers(0, k, n), base)
        labels[s] = rng.permutation(k)[noisy]
    for s, l in labels.items():
        l[:k] = np.arange(k)  # every cluster present
    J = jaccards(labels)
    orders, chi = {}, {}
    for s in labels:
        cand = [int(c) for c in rng.permutation(k)[:int(rng.integers(1, k+1))]]
        values = np.round(rng.uniform(0, 1, len(cand)), 1)  # coarse values give exact ties
        chi[s] = dict(zip(cand, map(float, values)))
        orders[s] = sorted(cand, key=lambda c: (-chi[s][c], c))
    picks, audit = pool.select_group(orders, J, chi)
    prev = audit['approved_rule_outcome']
    if prev['status'] == 'selected':
        assert picks == prev['selected'] and audit['status'] == 'selected'
    assert pool.one_line(audit)  # serialisable
