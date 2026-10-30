"""
ft8.py -- a small, readable FT8 encoder and waveform synthesizer (educational; written for the white paper).

Pipeline:  text -> 77-bit packed message -> +14-bit CRC (91 bits) -> LDPC(174,91) codeword
           -> Gray-coded 3-bit groups -> 58 data tones, interleaved with 3 x 7 Costas tones
           -> 79 tones -> continuous-phase Gaussian-filtered 8-FSK audio at 12 kHz.
"""
import numpy as np
from scipy.special import erf
import os

_here = os.path.dirname(os.path.abspath(__file__))
_t = np.load(os.path.join(_here, 'ldpc_tables.npz'))
G = _t['G'].astype(np.uint8)      # 83 x 91 parity generator
H = _t['H'].astype(np.uint8)      # 83 x 174 parity-check matrix
MN = _t['Mn']                     # 174 x 3  (1-origin check indices for each bit)

FS = 12000                        # audio sample rate used throughout
NSYM = 79
TSYM = 0.160                      # seconds per symbol
SPS = int(round(FS * TSYM))       # 1920 samples per symbol
TONE_SPACING = 1.0 / TSYM         # 6.25 Hz
COSTAS = (3, 1, 4, 0, 6, 5, 2)
GRAY = (0, 1, 3, 2, 5, 6, 4, 7)   # 3-bit value -> tone number
GRAY_INV = {t: b for b, t in enumerate(GRAY)}
SYNC_POS = (0, 36, 72)            # first symbol of each Costas block
DATA_POS = list(range(7, 36)) + list(range(43, 72))   # 58 data symbol positions

NTOKENS, MAX22, MAXGRID4 = 2063592, 4194304, 32400
A1 = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # 37 symbols
A2 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'    # 36
A3 = '0123456789'                              # 10
A4 = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'             # 27
FREE = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?'   # 42 characters for free text

# ----------------------------------------------------------------- source encoding (77 bits)
def int_to_bits(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]

def bits_to_int(b):
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v

def pack_basecall(call):
    """Standard callsign -> 28-bit-field offset (0 .. 262,177,559)."""
    c = call.upper()
    if len(c) > 6:
        raise ValueError('not a standard call: ' + call)
    if not c[2:3].isdigit():
        c = ' ' + c                      # digit must land in position 3 of the six-character template
    c = c.ljust(6)
    if c[2] not in A3:
        raise ValueError('callsign needs a digit in position 3: ' + call)
    i1, i2, i3 = A1.index(c[0]), A2.index(c[1]), A3.index(c[2])
    i4, i5, i6 = A4.index(c[3]), A4.index(c[4]), A4.index(c[5])
    return ((((i1 * 36 + i2) * 10 + i3) * 27 + i4) * 27 + i5) * 27 + i6

def pack28(call):
    """Return (n28, ip).  ip = 1 when the /R or /P suffix flag is set."""
    c = call.upper()
    if c == 'DE': return 0, 0
    if c == 'QRZ': return 1, 0
    if c == 'CQ': return 2, 0
    if c.startswith('CQ ') and len(c) < 8:          # CQ nnn  or  CQ a[bcd]
        mod = c[3:]
        if len(mod) == 3 and mod.isdigit(): return 3 + int(mod), 0
        if 1 <= len(mod) <= 4 and mod.isalpha():
            m = 0
            for ch in mod: m = 27 * m + (ord(ch) - 64)
            return 3 + 1000 + m, 0
        raise ValueError('bad CQ modifier')
    ip = 0
    if c.endswith('/R') or c.endswith('/P'):
        ip, c = 1, c[:-2]
    return NTOKENS + MAX22 + pack_basecall(c), ip

def packgrid(g):
    """Return (g15, ir).  Handles 4-char grids, +/-dd reports, R+/-dd, RRR, RR73, 73 and blank."""
    g = g.upper()
    if g == '': return MAXGRID4 + 1, 0
    if g == 'RRR': return MAXGRID4 + 2, 0
    if g == 'RR73': return MAXGRID4 + 3, 0
    if g == '73': return MAXGRID4 + 4, 0
    if len(g) == 4 and g[0] in 'ABCDEFGHIJKLMNOPQR' and g[1] in 'ABCDEFGHIJKLMNOPQR' and g[2:].isdigit():
        return ((( ord(g[0]) - 65) * 18 + (ord(g[1]) - 65)) * 10 + int(g[2])) * 10 + int(g[3]), 0
    ir = 0
    if g.startswith('R'):
        ir, g = 1, g[1:]
    return MAXGRID4 + 35 + int(g), ir

def pack_standard(call1, call2, extra):
    """Message type 1:  c28 | r1 | c28 | r1 | R1 | g15 | i3(=1)   -> 77 bits."""
    n1, p1 = pack28(call1)
    n2, p2 = pack28(call2)
    g15, ir = packgrid(extra)
    f = dict(c28a=n1, ipa=p1, c28b=n2, ipb=p2, R=ir, g15=g15, i3=1)
    bits = (int_to_bits(n1, 28) + [p1] + int_to_bits(n2, 28) + [p2] + [ir]
            + int_to_bits(g15, 15) + int_to_bits(1, 3))
    return bits, f

def pack_free(text):
    """Message type 0.0 (free text): 13 characters of base 42 -> 71 bits, then n3=0, i3=0."""
    t = text.upper()[:13].ljust(13)       # left-justified, padded with blanks on the right
    v = 0
    for ch in t:
        v = v * 42 + FREE.index(ch)
    assert v < 2 ** 71
    return int_to_bits(v, 71) + [0, 0, 0] + [0, 0, 0], dict(f71=v)

def pack_message(msg):
    """Pack a human-readable message.  Standard 'CALL1 CALL2 EXTRA' forms, 'CQ CALL GRID', or free text."""
    w = msg.upper().split()
    if len(w) == 4 and w[0] == 'CQ':                 # 'CQ DX CALL GRID' -> call1 = 'CQ DX'
        w = ['CQ ' + w[1]] + w[2:]
    if len(w) in (2, 3):
        try:
            return pack_standard(w[0], w[1], w[2] if len(w) == 3 else '')[0]
        except (ValueError, IndexError):
            pass
    return pack_free(msg)[0]

# ----------------------------------------------------------------- CRC-14 and LDPC
CRC_POLY = 0x2757                # x^14 + x^13 + x^10 + x^9 + x^8 + x^6 + x^4 + x^2 + x + 1 (0x6757 with leading term)

def crc14(bits77):
    """CRC of the 77-bit message zero-extended to 82 bits, polynomial 0x2757, initial value 0."""
    reg = 0
    for b in list(bits77) + [0] * 5:
        top = ((reg >> 13) & 1) ^ b
        reg = (reg << 1) & 0x3FFF
        if top:
            reg ^= CRC_POLY
    return reg

def ldpc_encode(bits91):
    u = np.array(bits91, dtype=np.uint8)
    parity = (G.astype(np.int64) @ u) % 2
    return np.concatenate([u, parity.astype(np.uint8)])

def encode_bits(bits77):
    """77 message bits -> (91 bits with CRC, 174-bit codeword)."""
    crc = crc14(bits77)
    b91 = list(bits77) + int_to_bits(crc, 14)
    return b91, ldpc_encode(b91)

def codeword_to_tones(cw):
    tones = np.zeros(NSYM, dtype=int)
    for s in SYNC_POS:
        tones[s:s + 7] = COSTAS
    for k, pos in enumerate(DATA_POS):
        v = bits_to_int(cw[3 * k:3 * k + 3])
        tones[pos] = GRAY[v]
    return tones

def message_to_tones(msg):
    b77 = pack_message(msg)
    b91, cw = encode_bits(b77)
    return codeword_to_tones(cw)

# ----------------------------------------------------------------- GFSK waveform
def gfsk_pulse(bt=2.0, sps=SPS):
    k = np.pi * np.sqrt(2.0 / np.log(2.0))
    t = np.arange(3 * sps) / sps - 1.5
    return (erf(k * bt * (t + 0.5)) - erf(k * bt * (t - 0.5))) / 2.0

def synth(tones, f0=1000.0, bt=2.0, fs=FS, return_phase=False):
    """Continuous-phase Gaussian-smoothed 8-FSK, modulation index 1 (tone spacing 1/T).
    Returns unit-amplitude real audio of 79*1920 samples (12.64 s)."""
    sps = int(round(fs * TSYM))
    n = len(tones)
    dphi_peak = 2 * np.pi / sps                       # radians/sample per tone step (h = 1)
    pulse = gfsk_pulse(bt, sps)
    dphi = np.full(n * sps + 2 * sps, 2 * np.pi * f0 / fs)
    for i, s in enumerate(tones):
        dphi[i * sps:i * sps + 3 * sps] += dphi_peak * s * pulse
    # dummy symbols before the first and after the last, equal to the first/last tone
    dphi[:2 * sps] += dphi_peak * pulse[sps:] * tones[0]
    dphi[n * sps:n * sps + 2 * sps] += dphi_peak * pulse[:2 * sps] * tones[-1]
    phi = np.concatenate([[0.0], np.cumsum(dphi[sps:sps + n * sps - 1])])
    x = np.sin(phi)
    nr = sps // 8                                     # 20 ms raised-cosine ramp on the very first/last samples
    ramp = (1 - np.cos(2 * np.pi * np.arange(nr) / (2 * nr))) / 2
    x[:nr] *= ramp
    x[-nr:] *= ramp[::-1]
    return (x, phi) if return_phase else x

def make_slot(tones, f0=1000.0, t0=0.5, slot=15.0, fs=FS, bt=2.0):
    """Place a transmission at t0 seconds into a 15 s slot (the normal FT8 timing)."""
    x = synth(tones, f0, bt, fs)
    out = np.zeros(int(slot * fs))
    i0 = int(round(t0 * fs))
    out[i0:i0 + len(x)] = x
    return out

def tones_to_text_bits(tones):
    """Inverse of the symbol mapping (hard decisions): tones -> 174-bit codeword."""
    cw = []
    for pos in DATA_POS:
        cw += int_to_bits(GRAY_INV[int(tones[pos])], 3)
    return np.array(cw, dtype=np.uint8)

if __name__ == '__main__':
    import sys
    m = ' '.join(sys.argv[1:]) or 'CQ W7TLG DN13'
    print(m, '->', ''.join(map(str, message_to_tones(m))))
