"""Mutual information of an 8-ary orthogonal (FSK) channel with uniform inputs in AWGN, in bits per tone
symbol, versus Es/N0.  'noncoherent' = unknown carrier phase (what an FT8 receiver actually has),
'coherent' = phase known.  Monte-Carlo with common random numbers."""
import numpy as np, json
from scipy.special import i0e
M = 8
rng = np.random.default_rng(11)
NS = 400000
n = (rng.normal(size=(NS, M)) + 1j * rng.normal(size=(NS, M))) / np.sqrt(2)     # CN(0,1)

def info_nc(rho):
    A = np.sqrt(rho)
    y = n.copy(); y[:, 0] += A
    z = 2 * A * np.abs(y)
    L = np.log(i0e(z)) + z
    return np.log2(M) - np.mean(np.log2(np.exp(L - L[:, [0]]).sum(1)))

def info_coh(rho):
    A = np.sqrt(rho)
    y = n.copy(); y[:, 0] += A
    L = 2 * A * y.real
    return np.log2(M) - np.mean(np.log2(np.exp(L - L[:, [0]]).sum(1)))

db = np.arange(-6, 16.01, 0.5)
nc = np.array([info_nc(10 ** (d / 10)) for d in db])
co = np.array([info_coh(10 ** (d / 10)) for d in db])

def invert(target, curve):
    return float(np.interp(target, curve, db))

out = dict(db=db.tolist(), nc=nc.tolist(), co=co.tolist())
for name, bits in (('91', 91), ('77', 77)):
    tgt = bits / 58.0
    out['esn0_nc_' + name] = invert(tgt, nc)
    out['esn0_co_' + name] = invert(tgt, co)
    out['snr_nc_' + name] = out['esn0_nc_' + name] - 10 * np.log10(400)
    out['snr_co_' + name] = out['esn0_co_' + name] - 10 * np.log10(400)
# ultimate (infinite bandwidth, Gaussian signalling): S/N0 = ln2 * bits / T
for bits in (91, 77):
    for T in (12.64, 15.0):
        sn0 = np.log(2) * bits / T
        out['ultimate_%d_%s' % (bits, T)] = 10 * np.log10(sn0 / 2500)
json.dump(out, open('capacity.json', 'w'), indent=1)
for k, v in out.items():
    if not isinstance(v, list): print(k, round(v, 2))
