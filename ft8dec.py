"""
ft8dec.py -- a compact FT8 receiver written for the white paper (educational, not a WSJT-X replacement).

  audio -> Costas sync search -> per-symbol 8-tone energies -> soft bit metrics (LLRs)
        -> sum-product belief propagation on the 83x174 LDPC graph -> CRC-14 check -> 77-bit message

No OSD, no a-priori (AP) decoding, no multi-symbol coherent combining, no signal subtraction.
"""
import numpy as np
from scipy.special import i0e
import ft8
from ft8 import FS, SPS, NSYM, COSTAS, SYNC_POS, DATA_POS, GRAY, GRAY_INV

H = ft8.H
# ---- sparse structure for belief propagation
_CHK = [np.nonzero(H[j])[0] for j in range(83)]
_PAD = np.full((83, 7), 174, dtype=int)          # index 174 = dummy variable
for j, v in enumerate(_CHK):
    _PAD[j, :len(v)] = v
_MASK = _PAD < 174
# for each variable, its three (check, slot) positions
_VPOS = np.zeros((174, 3, 2), dtype=int)
_cnt = np.zeros(174, dtype=int)
for j in range(83):
    for d, v in enumerate(_CHK[j]):
        _VPOS[v, _cnt[v]] = (j, d)
        _cnt[v] += 1
assert (_cnt == 3).all()

def parity_failures(bits):
    return int(((H.astype(np.int64) @ bits) % 2).sum())

def bp_decode(llr, max_iter=50, truth=None):
    """Sum-product decoding.  llr = log P(bit=0)/P(bit=1) for the 174 code bits.
    Returns (hard_bits, n_unsatisfied_checks, iterations_used, trace) where trace is a list of
    (unsatisfied checks, bit errors vs truth or None) per iteration."""
    llr = np.asarray(llr, float)
    llr_ext = np.concatenate([llr, [0.0]])
    v2c = llr_ext[_PAD].copy()
    trace = []
    hard = (llr < 0).astype(np.uint8)
    for it in range(1, max_iter + 1):
        t = np.tanh(np.clip(v2c, -30, 30) / 2.0)
        t[~_MASK] = 1.0
        # leave-one-out products
        pre = np.ones_like(t); suf = np.ones_like(t)
        for d in range(1, 7):
            pre[:, d] = pre[:, d - 1] * t[:, d - 1]
        for d in range(5, -1, -1):
            suf[:, d] = suf[:, d + 1] * t[:, d + 1]
        prod = np.clip(pre * suf, -0.9999999, 0.9999999)
        c2v = 2.0 * np.arctanh(prod)
        c2v[~_MASK] = 0.0
        post = llr + c2v[_VPOS[:, :, 0], _VPOS[:, :, 1]].sum(axis=1)
        hard = (post < 0).astype(np.uint8)
        nf = parity_failures(hard)
        trace.append((nf, None if truth is None else int((hard != truth).sum())))
        if nf == 0:
            return hard, 0, it, trace
        ext = np.concatenate([post, [0.0]])[_PAD] - c2v
        v2c = ext
    return hard, nf, max_iter, trace

# ------------------------------------------------------------------ front end
def tone_matrix(x, t0, f0):
    """Complex DFT values (79 x 8) at the 8 tone frequencies, for symbols starting at sample t0."""
    n = np.arange(SPS)
    k = np.arange(8)
    E = np.exp(-2j * np.pi * (f0 + k / ft8.TSYM)[:, None] * n[None, :] / FS)   # 8 x SPS
    S = np.zeros((NSYM, 8), complex)
    idx = t0 + np.arange(NSYM)[:, None] * SPS + n[None, :]
    ok = (idx[:, 0] >= 0) & (idx[:, -1] < len(x))
    seg = np.zeros((NSYM, SPS))
    seg[ok] = x[idx[ok]]
    S = seg @ E.T
    return S

def sync_metric(S):
    """Costas correlation: energy at the expected sync tones relative to the other tones, 21 symbols."""
    E = np.abs(S) ** 2
    num = den = 0.0
    for s in SYNC_POS:
        for k in range(7):
            e = E[s + k]
            num += e[COSTAS[k]]
            den += (e.sum() - e[COSTAS[k]]) / 7.0
    return num / max(den, 1e-30)

def spectrogram(x, hop=480, nfft=3840):
    """Waterfall: |FFT|^2 of one-symbol Hann windows, 2x zero padded (3.125 Hz bins)."""
    w = np.hanning(SPS)
    nfr = (len(x) - SPS) // hop + 1
    idx = np.arange(nfr)[:, None] * hop + np.arange(SPS)[None, :]
    F = np.fft.rfft(x[idx] * w[None, :], n=nfft, axis=1)
    return np.abs(F) ** 2

def coarse_sync(x, fmin=200.0, fmax=3000.0, tmin=-1.0, tmax=2.5, ncand=10, hop=480):
    """Costas search over time (40 ms steps) and frequency (3.125 Hz steps)."""
    P = spectrogram(x, hop)
    df = FS / 3840.0
    fb0, fb1 = int(fmin / df), int(fmax / df)
    t_lo, t_hi = int(round(tmin * FS / hop)), int(round(tmax * FS / hop))
    sps_hops = SPS // hop                      # 4 hops per symbol
    res = []
    for ft in range(fb0, fb1):
        for tt in range(t_lo, t_hi + 1):
            num = den = 0.0
            ok = True
            for s in SYNC_POS:
                for k in range(7):
                    row = tt + (s + k) * sps_hops
                    if row < 0 or row >= P.shape[0]:
                        ok = False; break
                    e = P[row, ft + 2 * COSTAS[k]]
                    num += e
                    den += P[row, ft:ft + 16:2].sum()
                if not ok: break
            if ok:
                res.append((num / max(den - num, 1e-30), tt * hop, ft * df))
    return res

def coarse_sync_fast(x, fmin=200.0, fmax=3000.0, tmin=-1.0, tmax=2.5, hop=480):
    """Vectorised version of coarse_sync; returns arrays (score[t,f], times, freqs)."""
    P = spectrogram(x, hop)
    df = FS / 3840.0
    fb0, fb1 = int(fmin / df), int(fmax / df)
    t_lo, t_hi = int(round(tmin * FS / hop)), int(round(tmax * FS / hop))
    sph = SPS // hop
    tts = np.arange(t_lo, t_hi + 1)
    nf = fb1 - fb0
    num = np.zeros((len(tts), nf)); den = np.zeros((len(tts), nf))
    for s in SYNC_POS:
        for k in range(7):
            rows = tts + (s + k) * sph
            valid = (rows >= 0) & (rows < P.shape[0])
            r = np.clip(rows, 0, P.shape[0] - 1)
            tone_cols = fb0 + 2 * COSTAS[k] + np.arange(nf)
            # total energy in the 8 tone bins for this candidate
            tot = np.zeros((len(tts), nf))
            for n in range(8):
                tot += P[r][:, fb0 + 2 * n: fb0 + 2 * n + nf] * valid[:, None]
            num += P[r][:, fb0 + 2 * COSTAS[k]: fb0 + 2 * COSTAS[k] + nf] * valid[:, None]
            den += tot
    score = num / np.maximum(den - num, 1e-30)
    # the number of valid rows differs near the edges; blank rows with any invalid symbol
    for s in SYNC_POS:
        for k in range(7):
            rows = tts + (s + k) * sph
            bad = (rows < 0) | (rows >= P.shape[0])
            score[bad, :] = 0
    return score, tts * hop / FS, (fb0 + np.arange(nf)) * df

_SYNC_SYMS = np.array([s + k for s in SYNC_POS for k in range(7)])
_SYNC_TONE = np.array([COSTAS[k] for s in SYNC_POS for k in range(7)])

def sync_energy(x, t0, f0):
    """Costas metric using only the 21 sync symbols (fast)."""
    n = np.arange(SPS)
    k = np.arange(8)
    E = np.exp(-2j * np.pi * (f0 + k / ft8.TSYM)[:, None] * n[None, :] / FS)
    idx = t0 + _SYNC_SYMS[:, None] * SPS + n[None, :]
    if idx.min() < 0 or idx.max() >= len(x):
        return 0.0
    en = np.abs(x[idx] @ E.T) ** 2                       # 21 x 8
    num = en[np.arange(21), _SYNC_TONE].sum()
    return num / max(en.sum() - num, 1e-30)

def fine_sync(x, t0, f0, dt=240, ddt=40, df=2.0, ddf=0.5):
    best = (-1, t0, f0)
    for a in range(-dt, dt + 1, ddt):
        for b in np.arange(-df, df + 1e-9, ddf):
            m = sync_energy(x, t0 + a, f0 + b)
            if m > best[0]:
                best = (m, t0 + a, f0 + b)
    return best

# ------------------------------------------------------------------ soft decisions
_BITS_OF_SYMBOL = np.array([[(v >> 2) & 1, (v >> 1) & 1, v & 1] for v in range(8)])   # bits of Gray *input* value
_TONE_BITS = np.array([_BITS_OF_SYMBOL[GRAY_INV[k]] for k in range(8)])                # tone -> 3 bits

def symbol_llrs(S, a_design=None):
    """Convert the 79x8 tone matrix to 174 LLRs (positive favours bit=0).
    Noise scale is estimated from the data; the signal strength from the 21 Costas symbols."""
    E = np.abs(S) ** 2
    m = np.median(E) / np.log(2.0)                 # exponential-noise mean energy estimate
    Y = np.sqrt(E / m)                              # amplitudes in units of the noise sigma
    if a_design is None:
        ec = np.mean([E[s + k, COSTAS[k]] for s in SYNC_POS for k in range(7)]) / m
        a2 = max(ec - 1.0, 0.25)
        a_design = np.sqrt(a2)
    llr = np.zeros(174)
    for i, pos in enumerate(DATA_POS):
        lk = np.log(i0e(2 * a_design * Y[pos])) + 2 * a_design * Y[pos]     # log I0
        for b in range(3):
            m0 = _TONE_BITS[:, b] == 0
            l0 = np.logaddexp.reduce(lk[m0]); l1 = np.logaddexp.reduce(lk[~m0])
            llr[3 * i + b] = l0 - l1
    return llr

def decode_at(x, t0, f0, max_iter=50, truth_bits=None):
    """Decode a candidate at known (t0 samples, f0 Hz).  Returns dict with status."""
    S = tone_matrix(x, t0, f0)
    llr = symbol_llrs(S)
    hard, nfail, its, trace = bp_decode(llr, max_iter, truth=truth_bits)
    out = dict(ok=False, bits91=None, iters=its, nfail=nfail, trace=trace, llr=llr, S=S)
    if nfail == 0:
        b91 = [int(v) for v in hard[:91]]
        if ft8.crc14(b91[:77]) == ft8.bits_to_int(b91[77:]):
            out.update(ok=True, bits91=b91, bits77=b91[:77])
    return out

# ------------------------------------------------------------------ unpacking (inverse of ft8.pack_*)
def _unpack_basecall(n):
    ch = []
    for alpha, base in ((ft8.A4, 27), (ft8.A4, 27), (ft8.A4, 27), (ft8.A3, 10), (ft8.A2, 36), (ft8.A1, 37)):
        n, r = divmod(n, base)
        ch.append(alpha[r])
    return ''.join(reversed(ch)).strip()

def unpack28(n):
    if n == 0: return 'DE'
    if n == 1: return 'QRZ'
    if n == 2: return 'CQ'
    if n < 1003: return 'CQ %03d' % (n - 3)
    if n < ft8.NTOKENS:
        m = n - 1003; s = ''
        while m: m, r = divmod(m, 27); s = chr(64 + r) + s
        return 'CQ ' + s
    if n < ft8.NTOKENS + ft8.MAX22: return '<hashed>'
    return _unpack_basecall(n - ft8.NTOKENS - ft8.MAX22)

def unpack77(b77):
    i3 = ft8.bits_to_int(b77[74:77])
    if i3 == 1 or i3 == 2:
        a = ft8.bits_to_int(b77[0:28]); pa = b77[28]
        b = ft8.bits_to_int(b77[29:57]); pb = b77[57]
        R = b77[58]; g = ft8.bits_to_int(b77[59:74])
        sfx = '/R' if i3 == 1 else '/P'
        c1 = unpack28(a) + (sfx if pa else ''); c2 = unpack28(b) + (sfx if pb else '')
        if g <= ft8.MAXGRID4:
            l = (g // 1800, (g // 100) % 18, (g // 10) % 10, g % 10)
            ex = chr(65 + g // 1800) + chr(65 + (g // 100) % 18) + str((g // 10) % 10) + str(g % 10)
            if R: ex = 'R ' + ex
        else:
            k = g - ft8.MAXGRID4
            ex = {1: '', 2: 'RRR', 3: 'RR73', 4: '73'}.get(k)
            if ex is None:
                ex = ('R' if R else '') + '%+03d' % (k - 35)
        return ' '.join(w for w in (c1, c2, ex) if w)
    if i3 == 0 and ft8.bits_to_int(b77[71:74]) == 0:
        v = ft8.bits_to_int(b77[:71]); s = ''
        for _ in range(13):
            v, r = divmod(v, 42); s = ft8.FREE[r] + s
        return s.strip()
    return '<unsupported type>'

# ------------------------------------------------------------------ channel
def add_noise(x, snr_db, rng, fs=FS):
    """Add white Gaussian noise so that  signal power / noise power in 2500 Hz = snr_db.
    'x' must be unit-amplitude (power 0.5 during transmission)."""
    sig_power = 0.5
    noise_power_2500 = sig_power / 10 ** (snr_db / 10.0)
    sigma2 = noise_power_2500 * (fs / 2.0) / 2500.0
    return x + rng.normal(0.0, np.sqrt(sigma2), len(x))

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    msg = 'CQ W7TLG DN13'
    tones = ft8.message_to_tones(msg)
    b77 = ft8.pack_message(msg)
    print('unpack77(pack) =', unpack77(b77))
    x = ft8.make_slot(tones, 1000.0, 0.5)
    for snr in (0, -10, -15, -18, -20, -22):
        y = add_noise(x, snr, rng)
        r = decode_at(y, 6000, 1000.0)
        print('SNR %4d dB  ok=%s  iters=%d  unsat=%d' % (snr, r['ok'], r['iters'], r['nfail']), unpack77(r['bits77']) if r['ok'] else '')
