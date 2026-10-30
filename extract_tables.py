"""Extract LDPC(174,91) tables from ft8_lib constants.c into a numpy .npz (read-only parse of untrusted text)."""
import re, sys, numpy as np
src = open(sys.argv[1]).read()
def block(name):
    m = re.search(name + r'[^=]*=\s*\{(.*?)\n\};', src, re.S)
    return m.group(1)
gen_txt = block(r'kFTX_LDPC_generator')
rows = re.findall(r'\{([^{}]*)\}', gen_txt)
G = np.array([[int(x,16) for x in r.replace(' ','').split(',') if x] for r in rows], dtype=np.uint8)
Gbits = np.unpackbits(G, axis=1)[:, :91]          # 83 x 91
nm_rows = re.findall(r'\{([^{}]*)\}', block(r'kFTX_LDPC_Nm'))
Nm = [[int(x) for x in r.replace(' ','').split(',') if x] for r in nm_rows]
nm_nums = [[v for v in r if v>0] for r in Nm]
mn_rows = re.findall(r'\{([^{}]*)\}', block(r'kFTX_LDPC_Mn'))
Mn = np.array([[int(x) for x in r.replace(' ','').split(',') if x] for r in mn_rows])
H = np.zeros((83,174), dtype=np.uint8)
for j,r in enumerate(nm_nums):
    for v in r: H[j, v-1] = 1
print('G', Gbits.shape, 'H', H.shape, 'row weights', sorted(set(H.sum(1))), 'col weights', sorted(set(H.sum(0))), 'Mn', Mn.shape)
np.savez('ldpc_tables.npz', G=Gbits, H=H, Mn=Mn)
# sanity: G rows give parity so that H c = 0 for random message
rng = np.random.default_rng(1)
for _ in range(5):
    u = rng.integers(0,2,91).astype(np.uint8)
    p = (Gbits @ u) % 2
    c = np.concatenate([u,p])
    assert not ((H @ c) % 2).any()
print('H c = 0 verified for random messages: encoder c=[u|G u] is consistent with parity-check matrix')
