"""Encode a message, bury it in noise, and see when the receiver can still decode it.
Run from the repository root:  python3 examples/01_encode_noise_decode.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import ft8, ft8dec

MESSAGE = "CQ W7TLG DN13"
tones = ft8.message_to_tones(MESSAGE)              # 79 tone numbers, 0..7
print("tones:", ''.join(map(str, tones)))
x = ft8.make_slot(tones, f0=1000.0, t0=0.5)        # 15 s of audio at 12 kHz, signal starts at 0.5 s
rng = np.random.default_rng(1)

print("SNR (dB in 2500 Hz)   decoded?")
for snr in (-16, -18, -19, -20, -21):
    y = ft8dec.add_noise(x, snr, rng)
    r = ft8dec.decode_at(y, t0=6000, f0=1000.0)    # start sample (0.5 s x 12000), tone-0 frequency
    text = ft8dec.unpack77(r['bits77']) if r['ok'] else '-'
    print(f"{snr:>10}            {r['ok']!s:<6} {text}")
