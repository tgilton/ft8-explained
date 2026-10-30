# ft8-explained

Companion code for the white paper **"How FT8 Really Works"** (Terry Gilton, W7TLG, 2026).
A small, readable FT8 encoder, waveform synthesizer and receiver in Python, used to generate every figure
and number in the paper. Educational code, not a WSJT-X replacement.

## The paper

**[Read the white paper (PDF)](docs/How_FT8_Really_Works.pdf)**: a plain-language explanation of the FT8 signal, the
15-second cycle, 77-bit message packing, the CRC, LDPC coding, Costas synchronization, and what "decoding at -20 dB"
means in light of Shannon's theorem. It is written for working hams; the quantitative details are in boxed
"Going deeper" sections and appendices. Every figure and number in it comes from the code in this repository.

| File | Purpose |
|---|---|
| `ft8.py` | Message packing (77 bits), CRC-14, LDPC(174,91) encoder, Gray map, Costas insertion, GFSK synthesis |
| `ft8dec.py` | Costas sync search, soft demapper, sum-product belief propagation, CRC check, unpacking |
| `sim_snr.py` | Monte-Carlo decode probability vs SNR (2500 Hz reference bandwidth), results in `sim_snr.json` |
| `capacity.py` | Mutual information of non-coherent / coherent 8-FSK and the SNR ladder, results in `capacity.json` |
| `figs_a.py`, `figs_b.py` | Figure scripts (output to `figs/`) |
| `extract_tables.py` | Builds `ldpc_tables.npz` from `ft8_lib`'s `constants.c` |
| `validate_encoder.sh` | Compares the encoder's 79 tones with the `ft8_lib` C reference (`gen_ft8`) |

## Quick start

    pip install numpy scipy matplotlib
    python3 ft8.py "CQ W7TLG DN13"      # prints the 79 tones
    python3 ft8dec.py                    # self-test: decode at several SNRs
    python3 sim_snr.py 250               # Monte Carlo (about 4 minutes on 2 cores)

## Validation and limits

For 13 test messages the 79-tone output matched `ft8_lib` exactly, the synthesized audio correlated 0.999999 with
the reference waveform, and the reference decoder decodes this audio. Not every message type is implemented
(no hashed or non-standard calls, DXpedition or contest types). The receiver omits OSD, multi-symbol detection,
a-priori decoding and signal subtraction, so it is about 1.3 dB less sensitive than WSJT-X. Noise is white Gaussian only.

`validate_encoder.sh` expects `ft8_lib` built at `/home/claude/ref/ft8_lib`; edit `REF=` to match your path.

## Credits

Protocol: Franke, Somerville and Taylor, "The FT4 and FT8 Communication Protocols," QEX, Jul/Aug 2020.
LDPC tables: [ft8_lib](https://github.com/kgoba/ft8_lib) by Kārlis Goba (MIT). Licensed MIT, see `LICENSE`.
