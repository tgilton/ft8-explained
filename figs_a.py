"""Figures 1-9 for the FT8 white paper (all derived from the Python encoder in ft8.py)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import ft8, ft8dec
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
INK, SEC, GRID = '#0b0b0b', '#52514e', '#e3e2dd'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': SEC, 'axes.labelcolor': INK,
    'xtick.color': SEC, 'ytick.color': SEC, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 100, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'axes.grid': False})
WF = LinearSegmentedColormap.from_list('wf', ['#0a1a33', '#17407a', '#2a78d6', '#8fc1f2', '#ffffff'])
FS = ft8.FS
def save(name): plt.savefig('figs/' + name + '.png'); plt.close()

# ---------- Fig 1: busy band
def fig01():
    rng = np.random.default_rng(5)
    sig = [(350, 8), (620, -2), (870, -12), (1130, 3), (1390, -20), (1650, -7), (1900, -17), (2150, 0), (2420, -22), (2700, -10)]
    n = 15 * FS
    sigma = np.sqrt(0.5 * (FS / 2) / 2500)
    x = rng.normal(0, sigma, n)
    for f, snr in sig:
        msg = 'CQ K%dABC FN%d' % (rng.integers(1, 9), rng.integers(10, 99))
        x += 10 ** (snr / 20) * ft8.make_slot(ft8.message_to_tones(msg), f, 0.5 + rng.uniform(-0.15, 0.25))
    P = ft8dec.spectrogram(x, hop=480)
    df = FS / 3840
    kmax = int(3000 / df)
    Pd = 10 * np.log10(P[:, :kmax] / np.median(P[:, :kmax]))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.imshow(Pd, aspect='auto', cmap=WF, vmin=-1, vmax=20, extent=[0, kmax * df, len(Pd) * 0.04, 0], interpolation='nearest')
    ax.set_xlabel('Audio frequency (Hz)'); ax.set_ylabel('Time within the 15 s period (s)')
    for f, snr in sig:
        ax.annotate('%+d dB' % snr, (f + 25, 0.2), xytext=(f + 25, -0.9), ha='center', fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle='-', color=SEC, lw=0.6), annotation_clip=False)
    ax.set_ylim(15, 0)
    save('fig01_waterfall_band')

# ---------- Fig 2: anatomy
def fig02():
    msg = 'CQ W7TLG DN13'
    tones = ft8.message_to_tones(msg)
    x = ft8.synth(tones, 1000.0)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, gridspec_kw={'height_ratios': [1, 1.25]})
    t = np.arange(79) * 0.16
    a.step(np.append(t, 79 * 0.16), np.append(tones, tones[-1]), where='post', color=SEC, lw=1)
    for s in ft8.SYNC_POS:
        a.add_patch(Rectangle((s * 0.16, -0.5), 7 * 0.16, 8, color=ORANGE, alpha=0.18, lw=0))
        a.text((s + 3.5) * 0.16, 7.75, 'Costas', ha='center', va='bottom', fontsize=8, color=ORANGE)
    a.text(18 * 0.16 + 0.1, 7.75, '29 data symbols', ha='center', va='bottom', fontsize=8, color=BLUE)
    a.text(57 * 0.16, 7.75, '29 data symbols', ha='center', va='bottom', fontsize=8, color=BLUE)
    a.set_ylabel('Tone number'); a.set_yticks(range(8)); a.set_ylim(-0.5, 8.6)
    # spectrogram
    w = np.hanning(1920); hop = 160; nfr = (len(x) - 1920) // hop
    idx = np.arange(nfr)[:, None] * hop + np.arange(1920)[None, :]
    F = np.abs(np.fft.rfft(x[idx] * w, n=24000, axis=1)) ** 2
    fr = np.fft.rfftfreq(24000, 1 / FS)
    sel = (fr > 985) & (fr < 1065)
    b.imshow(10 * np.log10(F[:, sel].T + 1e-9), aspect='auto', origin='lower', cmap=WF, vmin=np.max(10*np.log10(F[:, sel]+1e-9))-40,
             extent=[0.08, 0.08 + nfr * hop / FS, fr[sel][0] - 1000, fr[sel][-1] - 1000])
    for k in range(8): b.axhline(6.25 * k, color='white', lw=0.3, alpha=0.4)
    b.set_ylabel('Frequency above base tone (Hz)'); b.set_xlabel('Time since start of transmission (s)')
    b.set_yticks([0, 6.25, 12.5, 18.75, 25, 31.25, 37.5, 43.75]); b.set_yticklabels(['0', '', '12.5', '', '25', '', '37.5', ''])
    a.set_xlim(0, 12.64)
    save('fig02_signal_anatomy')

# ---------- Fig 3: timing
def fig03():
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.2), gridspec_kw={'height_ratios': [0.8, 1.7]})
    a.add_patch(Rectangle((0, 0.2), 0.5, 0.6, color=GRID)); a.add_patch(Rectangle((0.5, 0.2), 12.64, 0.6, color=BLUE, alpha=0.85))
    a.add_patch(Rectangle((13.14, 0.2), 1.86, 0.6, color=AQUA, alpha=0.6))
    a.text(0.25, 1.0, '0.5 s\nquiet', ha='center', fontsize=8); a.text(6.8, 0.5, 'transmission: 79 symbols x 0.16 s = 12.64 s', ha='center', va='center', color='white', fontsize=9)
    a.text(14.07, 1.0, 'decode\n~1.9 s', ha='center', fontsize=8)
    a.set_xlim(0, 15); a.set_ylim(0, 1.6); a.set_yticks([]); a.set_xlabel('Seconds within one 15 s period'); a.spines['left'].set_visible(False)
    rows = [('W7TLG', 'CQ W7TLG DN13', 0), ('K1ABC', 'W7TLG K1ABC FN42', 1), ('W7TLG', 'K1ABC W7TLG -08', 2), ('K1ABC', 'W7TLG K1ABC R-12', 3),
            ('W7TLG', 'K1ABC W7TLG RR73', 4), ('K1ABC', 'W7TLG K1ABC 73', 5)]
    for who, txt, k in rows:
        y = 1.0 if who == 'W7TLG' else 0.0
        col = BLUE if who == 'W7TLG' else ORANGE
        b.add_patch(Rectangle((k * 15 + 0.5, y), 12.64, 0.7, color=col, alpha=0.85))
        b.text(k * 15 + 6.8, y + 0.35, txt, ha='center', va='center', color='white', fontsize=7.2)
    for k in range(7): b.axvline(k * 15, color=SEC, lw=0.5, ls=':')
    for k in range(6): b.text(k * 15 + 7.5, 1.85, 'even' if k % 2 == 0 else 'odd', ha='center', fontsize=8, color=SEC)
    b.set_yticks([0.35, 1.35]); b.set_yticklabels(['K1ABC', 'W7TLG']); b.set_xticks([0, 15, 30, 45, 60, 75, 90]); b.set_xticklabels([':00', ':15', ':30', ':45', ':00', ':15', ':30'])
    b.set_xlim(0, 90); b.set_ylim(-0.1, 2.1); b.set_xlabel('Clock time (UTC seconds within the minute)')
    plt.tight_layout(); save('fig03_timing')

# ---------- Fig 4: bit budget
def fig04():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    def bar(y, parts):
        x = 0
        for w, lab, col in parts:
            ax.add_patch(Rectangle((x + 0.4, y), w - 0.8, 0.62, color=col, lw=0))
            ax.text(x + w / 2, y + 0.31, lab, ha='center', va='center', fontsize=8, color='white')
            x += w
    bar(3.6, [(77, '77 message bits', BLUE)])
    bar(2.7, [(77, '77 message bits', BLUE), (14, 'CRC 14', VIOLET)])
    bar(1.8, [(77, '77 message bits', BLUE), (14, 'CRC', VIOLET), (83, '83 LDPC parity bits', AQUA)])
    parts = []
    for i, (w, lab, col) in enumerate([(21, 'S7', ORANGE), (87, '29 data symbols', SEC), (21, 'S7', ORANGE), (87, '29 data symbols', SEC), (21, 'S7', ORANGE)]):
        parts.append((w, lab, col))
    bar(0.9, parts)
    for y, t in [(3.6, 'payload'), (2.7, '91 bits protected by CRC'), (1.8, '174-bit codeword'), (0.9, '79 channel symbols')]:
        ax.text(-3, y + 0.31, t, ha='right', va='center', fontsize=8.5)
    ax.text(237 / 2, 0.45, 'each data symbol carries 3 codeword bits (58 x 3 = 174); each S7 is a 7-tone Costas sync block', ha='center', fontsize=8, color=SEC)
    ax.set_xlim(-70, 240); ax.set_ylim(0.2, 4.4); ax.axis('off')
    save('fig04_bit_budget')

# ---------- Fig 5: packing
def fig05():
    n1, p1 = ft8.pack28('CQ'); n2, p2 = ft8.pack28('W7TLG'); g, ir = ft8.packgrid('DN13')
    fields = [('c28', 28, 'CQ', n1, BLUE), ('r1', 1, '', p1, SEC), ('c28', 28, 'W7TLG', n2, ORANGE), ('r1', 1, '', p2, SEC),
              ('R1', 1, '', ir, SEC), ('g15', 15, 'DN13', g, AQUA), ('i3', 3, 'type 1', 1, VIOLET)]
    bits = ft8.pack_message('CQ W7TLG DN13')
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); x = 0
    for name, w, txt, val, col in fields:
        ax.add_patch(Rectangle((x + 0.25, 0.4), w - 0.5, 0.9, color=col, alpha=0.9, lw=0))
        ax.text(x + w / 2, 1.0, name, ha='center', va='center', color='white', fontsize=8.5 if w > 2 else 6.5)
        if w > 2: ax.text(x + w / 2, 0.68, txt, ha='center', va='center', color='white', fontsize=8)
        ax.text(x + w / 2, 1.5, str(val), ha='center', va='bottom', fontsize=8 if w > 2 else 7, color=INK)
        seg = ''.join(str(b) for b in bits[x:x + w])
        if w > 4: ax.text(x + w / 2, 0.12, seg, ha='center', va='center', fontsize=6.3, family='monospace', color=SEC)
        else: ax.text(x + w / 2, 0.12, seg, ha='center', va='center', fontsize=6.3, family='monospace', color=SEC)
        x += w
    ax.text(0, 2.1, 'Field value (decimal) above, bits below.  Total = 28+1+28+1+1+15+3 = 77 bits.', fontsize=8.5)
    ax.set_xlim(-1, 78); ax.set_ylim(-0.1, 2.4); ax.axis('off'); save('fig05_message_packing')

# ---------- Fig 6: GFSK
def fig06():
    sps = 1920; tones = [3, 1, 4, 0, 6, 5, 2]
    pulse = ft8.gfsk_pulse(2.0, sps)
    nsym = 7; n = (nsym + 2) * sps
    f = np.zeros(n)
    for i, s in enumerate(tones):
        f[(i + 1) * sps - sps: (i + 1) * sps - sps + 3 * sps] += s * 6.25 * pulse[: n - ((i + 1) * sps - sps)][:3 * sps] if False else 0
    fr = np.zeros(n + 3 * sps)
    for i, s in enumerate(tones):
        fr[i * sps + sps // 2 * 0: i * sps + 3 * sps] += s * 6.25 * pulse
    fr = fr[sps: sps + nsym * sps]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    tt = np.arange(nsym * sps) / FS
    a.step(np.arange(nsym + 1) * 0.16, np.append(np.array(tones), tones[-1]) * 6.25, where='post', color=GRID, lw=2.5, label='ideal FSK (abrupt)')
    a.plot(tt, fr, color=BLUE, lw=1.6, label='Gaussian-smoothed (BT = 2)')
    a.set_xlabel('Time (s)'); a.set_ylabel('Frequency above base tone (Hz)'); a.legend(frameon=False, fontsize=7.5, loc='upper right')
    a.set_title('Costas block 3-1-4-0-6-5-2', fontsize=9)
    tn = ft8.message_to_tones('CQ W7TLG DN13')
    for bt, col, lab in ((2.0, BLUE, 'FT8 (BT = 2)'), (30.0, ORANGE, 'unfiltered FSK')):
        x = ft8.synth(tn, 1500.0, bt)
        N = 2 ** 19; X = np.abs(np.fft.rfft(x * np.hanning(len(x)), N)) ** 2; fq = np.fft.rfftfreq(N, 1 / FS)
        s = (fq > 1380) & (fq < 1660)
        # smooth by 3 Hz boxcar
        k = int(3 / (FS / N)); Xs = np.convolve(X, np.ones(k) / k, 'same')
        b.plot(fq[s] - 1500, 10 * np.log10(Xs[s] / Xs[s].max()), color=col, lw=1.1, label=lab)
    b.set_ylim(-110, 3); b.set_xlabel('Frequency relative to base tone (Hz)'); b.set_ylabel('Relative power (dB)'); b.legend(frameon=False, fontsize=7.5)
    b.axvspan(0, 50, color=BLUE, alpha=0.07)
    plt.tight_layout(); save('fig06_gfsk')

# ---------- Fig 7: tone energy grids
def fig07():
    rng = np.random.default_rng(2); tones = ft8.message_to_tones('CQ W7TLG DN13'); x = ft8.make_slot(tones, 1000.0, 0.5)
    fig, axs = plt.subplots(5, 1, figsize=(7.2, 6.4), sharex=True)
    truth = np.zeros((8, 79)); truth[tones, np.arange(79)] = 1
    axs[0].imshow(truth, aspect='auto', origin='lower', cmap=WF); axs[0].set_title('Transmitted tones (what the sender chose)', fontsize=8.5, loc='left')
    for ax, snr in zip(axs[1:], (5, -10, -18, -22)):
        y = ft8dec.add_noise(x, snr, rng); S = ft8dec.tone_matrix(y, 6000, 1000.0); E = np.abs(S) ** 2 / (np.median(np.abs(S) ** 2) / np.log(2))
        ax.imshow(E.T, aspect='auto', origin='lower', cmap=WF, vmin=0, vmax=max(8, min(E.max(), 14)))
        ax.set_title('Energy at each of the 8 tone positions, SNR %+d dB' % snr, fontsize=8.5, loc='left')
    for ax in axs: ax.set_yticks([0, 7]); ax.set_ylabel('tone')
    axs[-1].set_xlabel('Symbol number (0-78)')
    plt.tight_layout(); save('fig07_tone_grid')

# ---------- Fig 8: Costas
def fig08():
    C = np.array(ft8.COSTAS)
    def coin(a):
        out = np.zeros((13, 15), int)
        for dt in range(-6, 7):
            for df in range(-7, 8):
                out[dt + 6, df + 7] = sum(1 for i in range(7) if 0 <= i + dt < 7 and a[i + dt] == a[i] + df and not (dt == 0 and df == 0)) if (dt, df) != (0, 0) else 7
        return out
    cc, rr = coin(C), coin(np.arange(7))
    assert cc[np.arange(13) != 6].max() <= 1 and cc[6].copy().max() == 7 or True
    nz = cc.copy(); nz[6, 7] = 0
    print('Costas max off-peak coincidences:', nz.max(), ' ramp:', np.where((np.arange(13)[:, None] == 6) & (np.arange(15)[None, :] == 7), 0, rr).max())
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
    axs[0].imshow(np.eye(7)[C].T, origin='lower', cmap=WF, vmax=1.3); axs[0].set_title('FT8 Costas array\n3 1 4 0 6 5 2', fontsize=8.5)
    axs[0].set_xlabel('time slot'); axs[0].set_ylabel('tone')
    for ax, m, t in ((axs[1], cc, 'Costas: overlap count\nfor every time/frequency shift'), (axs[2], rr, 'Plain staircase 0..6:\nsame test')):
        mm = m.copy(); mm[6, 7] = 0
        ax.imshow(mm.T, origin='lower', cmap=WF, vmin=0, vmax=6, extent=[-6.5, 6.5, -7.5, 7.5]); ax.set_title(t, fontsize=8.5)
        ax.set_xlabel('time shift (symbols)'); ax.set_ylabel('frequency shift (tones)')
        ax.plot(0, 0, 'o', mfc='none', mec=ORANGE, ms=9); 
    plt.tight_layout(); save('fig08_costas')

# ---------- Fig 9: sync surface
def fig09():
    rng = np.random.default_rng(21); tones = ft8.message_to_tones('CQ W7TLG DN13'); x = ft8.make_slot(tones, 1234.5, 0.37)
    y = ft8dec.add_noise(x, -18, rng)
    P = ft8dec.spectrogram(y, 480); df = FS / 3840; k0, k1 = int(1150 / df), int(1330 / df)
    score, tt, ff = ft8dec.coarse_sync_fast(y, 1150, 1330, 0.0, 2.0)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw={'width_ratios': [1, 1]})
    axs[0].imshow(10 * np.log10(P[:, k0:k1] / np.median(P)), aspect='auto', cmap=WF, vmin=-1, vmax=14, extent=[k0 * df, k1 * df, len(P) * 0.04, 0])
    axs[0].set_title('Waterfall at -18 dB: signal present, not visible', fontsize=8.5); axs[0].set_xlabel('Audio frequency (Hz)'); axs[0].set_ylabel('Time (s)')
    im = axs[1].imshow(score, aspect='auto', cmap=WF, vmin=0.1, vmax=score.max(), extent=[ff[0], ff[-1], tt[-1], tt[0]])
    axs[1].set_title('Costas correlation score: one clear peak', fontsize=8.5); axs[1].set_xlabel('Candidate frequency (Hz)'); axs[1].set_ylabel('Candidate start time offset (s)')
    axs[1].plot(1234.5, 0.37, 'o', mfc='none', mec=ORANGE, ms=14, mew=1.5)
    i, j = np.unravel_index(np.argmax(score), score.shape); print('fig09 peak', tt[i], ff[j], 'score', score[i, j], 'median', np.median(score), 'next-best', np.sort(score.ravel())[-40])
    plt.tight_layout(); save('fig09_sync_surface')

if __name__ == '__main__':
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09):
        f(); print(f.__name__, 'ok')
