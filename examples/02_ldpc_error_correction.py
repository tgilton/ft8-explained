"""Watch the LDPC(174,91) code repair wrong bits.  Run from the repository root:
python3 examples/02_ldpc_error_correction.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import ft8, ft8dec

rng = np.random.default_rng(2)
b77 = ft8.pack_message("CQ W7TLG DN13")
b91, codeword = ft8.encode_bits(b77)               # 174 bits
WRONG_STRENGTH = 1.5                               # try 3.0: confidently wrong bits are much harder to fix

print("wrong bits   result")
for n_wrong in (5, 10, 15, 20, 25, 30):
    wins = 0
    for trial in range(20):
        llr = np.where(codeword == 0, 4.0, -4.0)   # LLR > 0 means "bit is 0"; start with correct beliefs
        bad = rng.choice(174, n_wrong, replace=False)
        llr[bad] = -WRONG_STRENGTH * np.sign(llr[bad])   # these bits are believed wrong
        hard, unsatisfied, iters, _ = ft8dec.bp_decode(llr)
        wins += (unsatisfied == 0 and (hard == codeword).all())
    print(f"{n_wrong:>8}     corrected in {wins}/20 trials")
