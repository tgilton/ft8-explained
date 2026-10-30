"""Monte-Carlo: probability of a correct decode versus SNR (2500 Hz reference bandwidth), AWGN only.
Two receivers:  'ideal'  = time and frequency known exactly;  'sync' = Costas search (coarse + fine) first."""
import numpy as np, json, sys, time
from multiprocessing import Pool
import ft8, ft8dec

def trial(args):
    snr, mode, seed = args
    rng = np.random.default_rng(seed)
    b77 = [int(v) for v in rng.integers(0, 2, 77)]
    b77[74:77] = [0, 0, 1]                        # i3 = 1 (irrelevant to the decoder, keeps it a legal type)
    b91, cw = ft8.encode_bits(b77)
    tones = ft8.codeword_to_tones(cw)
    f_true = rng.uniform(800, 2500)
    t_true = rng.uniform(0.3, 0.9)
    x = ft8.make_slot(tones, f_true, t_true)
    y = ft8dec.add_noise(x, snr, rng)
    if mode == 'ideal':
        t0, f0 = int(round(t_true * ft8.FS)), f_true
    else:
        score, tt, ff = ft8dec.coarse_sync_fast(y, fmin=f_true - 50, fmax=f_true + 50, tmin=-0.5, tmax=1.5)
        i, j = np.unravel_index(np.argmax(score), score.shape)
        _, t0, f0 = ft8dec.fine_sync(y, int(round(tt[i] * ft8.FS)), ff[j])
    r = ft8dec.decode_at(y, t0, f0, max_iter=50)
    good = r['ok'] and r['bits77'] == b77
    bad = r['ok'] and not good
    return (snr, mode, int(good), int(bad), r['iters'])

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    snrs = list(range(-25, -13))
    jobs = [(s, m, 1000 * (s + 40) + k + (0 if m == 'ideal' else 500000)) for s in snrs for m in ('ideal', 'sync') for k in range(N)]
    t = time.time()
    with Pool(2) as p:
        res = p.map(trial, jobs, chunksize=10)
    out = {}
    for snr, mode, good, bad, it in res:
        d = out.setdefault(mode, {}).setdefault(str(snr), dict(n=0, good=0, bad=0, iters=[]))
        d['n'] += 1; d['good'] += good; d['bad'] += bad
        if good: d['iters'].append(it)
    for m in out:
        for s in out[m]:
            out[m][s]['mean_iters'] = float(np.mean(out[m][s]['iters'])) if out[m][s]['iters'] else None
            del out[m][s]['iters']
    json.dump(out, open('sim_snr.json', 'w'), indent=1)
    print('done in %.0fs' % (time.time() - t))
    for m in out:
        print(m, {s: round(v['good'] / v['n'], 2) for s, v in out[m].items()})
