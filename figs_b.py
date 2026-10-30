import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import ft8, ft8dec
from figs_a import BLUE, ORANGE, AQUA, VIOLET, INK, SEC, GRID, WF, save
FS = ft8.FS

def fig10():
    Hh = np.array([[1,1,0,1,1,0,0],[1,0,1,1,0,1,0],[0,1,1,1,0,0,1]])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={'width_ratios': [1, 1.25]})
    for i in range(7):
        for j in range(3):
            if Hh[j, i]: a.plot([i, 1 + 2 * j], [1, 0], color=GRID if False else '#b8b7b0', lw=1, zorder=1)
    for i in range(7):
        a.add_patch(Circle((i, 1), 0.22, color=BLUE, zorder=2)); a.text(i, 1, 'b%d' % (i + 1), ha='center', va='center', color='white', fontsize=7, zorder=3)
    for j in range(3):
        a.add_patch(Rectangle((1 + 2 * j - 0.2, -0.2), 0.4, 0.4, color=ORANGE, zorder=2)); a.text(1 + 2 * j, 0, 'c%d' % (j + 1), ha='center', va='center', color='white', fontsize=7, zorder=3)
    a.text(3, 1.5, 'bits (variable nodes)', ha='center', fontsize=8, color=BLUE); a.text(3, -0.6, 'parity rules (check nodes)', ha='center', fontsize=8, color=ORANGE)
    a.set_xlim(-0.6, 6.6); a.set_ylim(-0.9, 1.8); a.set_aspect('equal'); a.axis('off'); a.set_title('A toy code: 7 bits, 3 rules', fontsize=9)
    H = ft8.H
    ys, xs = np.nonzero(H)
    b.scatter(xs, ys, s=1.6, color=BLUE, marker='s', linewidths=0)
    b.set_xlim(-1, 174); b.set_ylim(83, -1); b.set_xlabel('codeword bit (0-173)'); b.set_ylabel('parity rule (0-82)')
    b.set_title('FT8 code: 522 ones in 14,442 cells (3.6%)', fontsize=9, pad=22)
    b.axvline(90.5, color=ORANGE, lw=0.8); b.text(45, -4.5, '91 message+CRC bits', ha='center', fontsize=7.5, color=SEC, clip_on=False); b.text(132, -4.5, '83 parity bits', ha='center', fontsize=7.5, color=SEC, clip_on=False)
    plt.tight_layout(); save('fig10_ldpc')

def fig11():
    snr = -19.0; out = []
    for k in range(14):
        rng = np.random.default_rng(100 + k)
        b77 = [int(v) for v in rng.integers(0, 2, 77)]; b91, cw = ft8.encode_bits(b77); tones = ft8.codeword_to_tones(cw)
        y = ft8dec.add_noise(ft8.make_slot(tones, 1000.0, 0.5), snr, rng)
        S = ft8dec.tone_matrix(y, 6000, 1000.0); llr = ft8dec.symbol_llrs(S)
        hard, nf, its, tr = ft8dec.bp_decode(llr, 50, truth=cw)
        out.append((nf == 0, tr, int(((llr < 0).astype(int) != cw).sum())))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ok, tr, e0 in out:
        col = BLUE if ok else ORANGE
        a.plot(range(1, len(tr) + 1), [t[0] for t in tr], color=col, lw=1.1, alpha=0.9)
        b.plot(range(1, len(tr) + 1), [t[1] for t in tr], color=col, lw=1.1, alpha=0.9)
    a.set_xlabel('BP iteration'); a.set_ylabel('Parity rules still violated (of 83)'); b.set_xlabel('BP iteration'); b.set_ylabel('Wrong code bits (of 174)')
    a.plot([], [], color=BLUE, label='decoded (CRC ok)'); a.plot([], [], color=ORANGE, label='failed after 50 iterations'); a.legend(frameon=False, fontsize=7.5)
    a.set_title('Frames at -19 dB', fontsize=9, loc='left')
    plt.tight_layout(); save('fig11_bp_convergence')
    print('fig11: first-pass bit errors', [o[2] for o in out], 'ok', [o[0] for o in out])

def fig12():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    items = [('Noise in 2500 Hz (the reference bandwidth)', 0, GRID, 'SNR = -20 dB'), ('Noise in 50 Hz (the whole FT8 signal)', -17, '#b8b7b0', 'SNR = -3 dB'),
             ('Noise in 6.25 Hz (one tone)', -26, SEC, 'SNR = +6 dB')]
    for i, (lab, v, c, snr) in enumerate(items):
        y = i * 1.0
        ax.barh(y, v + 30, left=-30, color=c, height=0.42)
        ax.text(-29.7, y - 0.33, lab, va='bottom', fontsize=8.5, color=INK)
        ax.text(v + 0.5 if v < 0 else v - 0.5, y, '%d dB' % v, va='center', ha='left' if v < 0 else 'right', fontsize=8.5)
        ax.text(9.5, y, snr, va='center', ha='right', fontsize=9, color=BLUE, fontweight='bold')
    ax.axvline(-20, color=BLUE, lw=2); ax.text(-20, 2.75, 'signal power = -20 dB', color=BLUE, ha='center', fontsize=8.5)
    ax.set_xlim(-30, 9.8); ax.set_ylim(2.95, -0.7); ax.set_yticks([]); ax.set_xlabel('Power relative to the noise in 2500 Hz (dB)'); ax.spines['left'].set_visible(False)
    plt.tight_layout(); save('fig12_snr_bandwidth')

def fig13():
    c = json.load(open('capacity.json')); sim = json.load(open('sim_snr.json'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={'width_ratios': [1.1, 1]})
    s = np.linspace(-32, -8, 200); lin = 10 ** (s / 10)
    a.plot(s, 1.4427 * 2500 * lin, color=SEC, lw=1.4, label='Shannon limit (any signalling)')
    db = np.array(c['db']); nc = np.array(c['nc']); co = np.array(c['co'])
    a.plot(db - 10 * np.log10(400), 58 * nc / 12.64, color=BLUE, lw=1.4, label='8-FSK, phase unknown')
    a.plot(db - 10 * np.log10(400), 58 * co / 12.64, color=AQUA, lw=1.4, label='8-FSK, phase known')
    a.plot([-20.8], [91 / 12.64], 'o', color=ORANGE, ms=7, label='WSJT-X, 50 % decode (QEX Table 6)')
    a.set_yscale('log'); a.set_xlim(-30, -12); a.set_ylim(1, 100); a.set_xlabel('SNR in 2500 Hz (dB)'); a.set_ylabel('Information rate (bits/s)')
    a.legend(frameon=False, fontsize=7, loc='upper left')
    pts = [('Shannon, 91 bits in 12.64 s', c['ultimate_91_12.64'], SEC), ('8-FSK, phase known', c['snr_co_91'], AQUA), ('8-FSK, phase unknown', c['snr_nc_91'], BLUE),
           ('WSJT-X FT8 (published)', -20.8, ORANGE)]
    d = sim['ideal']; snrs = sorted(int(k) for k in d); pr = [d[str(k)]['good'] / d[str(k)]['n'] for k in snrs]
    thr = float(np.interp(0.5, pr[snrs.index(-21):snrs.index(-17)+1], snrs[snrs.index(-21):snrs.index(-17)+1]))
    pts.append(('This paper\'s simple receiver', thr, VIOLET)); print('BP-only 50% threshold (ideal sync):', thr)
    for i, (lab, v, col) in enumerate(pts):
        b.plot([v], [i], 'o', color=col, ms=7); b.text(v + 0.35, i, '%.1f dB  %s' % (v, lab), va='center', fontsize=7.5)
    b.set_xlim(-29, -4); b.set_ylim(-0.6, len(pts) - 0.4); b.invert_yaxis(); b.set_yticks([]); b.set_xlabel('SNR in 2500 Hz needed for 50 % success (dB)'); b.spines['left'].set_visible(False)
    plt.tight_layout(); save('fig13_shannon'); return thr

def fig14():
    sim = json.load(open('sim_snr.json')); fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.axvspan(-24.5, -21, color='#e3e2dd', alpha=0.6, lw=0); ax.axvspan(-21, -18, color=ORANGE, alpha=0.13, lw=0); ax.axvspan(-18, -14.5, color=BLUE, alpha=0.08, lw=0)
    for x, lab in ((-22.75, 'out of reach'), (-19.5, 'every dB counts'), (-16.25, 'decodes anyway')):
        ax.text(x, 1.085, lab, ha='center', va='bottom', fontsize=8, color=INK)
    for mode, col, lab in (('ideal', BLUE, 'time and frequency known'), ('sync', ORANGE, 'with Costas search')):
        d = sim[mode]; s = sorted(int(k) for k in d); p = [d[str(k)]['good'] / d[str(k)]['n'] for k in s]
        ax.plot(s, p, 'o-', color=col, lw=1.4, ms=4, label=lab)
    ax.axvline(-20.8, color=SEC, ls='--', lw=1); ax.text(-20.9, 0.55, 'WSJT-X published\n50% point (-20.8 dB)', ha='right', fontsize=7.5, color=SEC)
    ax.set_xlim(-24.5, -14.5); ax.set_ylim(-0.03, 1.05); ax.set_xlabel('SNR in 2500 Hz (dB)'); ax.set_ylabel('Probability of correct decode'); ax.legend(frameon=False, fontsize=8, loc='lower right'); ax.grid(color=GRID, lw=0.6)
    plt.tight_layout(); save('fig14_decode_probability')

if __name__ == '__main__':
    for f in (fig10, fig11, fig12, fig13, fig14): f(); print(f.__name__, 'ok')
