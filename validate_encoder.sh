#!/bin/bash
# Compare the Python encoder with the independent C reference (ft8_lib gen_ft8) on several messages.
REF=${FT8_LIB:-$HOME/ft8_lib}   # path to a built ft8_lib checkout (needs ./gen_ft8); override with FT8_LIB=/path
fail=0
while IFS= read -r msg; do
  ref=$($REF/gen_ft8 "$msg" /tmp/ref_$$.wav 2>&1 | grep "FSK tones" | awk '{print $3}')
  me=$(python3 -I ft8.py "$msg" | awk -F'-> ' '{print $2}')
  if [ "$ref" == "$me" ]; then echo "OK   $msg"; else echo "FAIL $msg"; echo " ref $ref"; echo " py  $me"; fail=1; fi
done <<'MSGS'
CQ W7TLG DN13
CQ K1ABC FN42
W7TLG K1ABC -12
K1ABC W7TLG R-07
W7TLG K1ABC RR73
K1ABC W7TLG 73
W7TLG JA1XYZ PM95
CQ DX JA1XYZ PM95
G4WJS K1JT +03
TNX BOB 73 GL
5W SOLAR PWR
VE3ABC W7TLG R+10
K1ABC/R W7TLG -03
MSGS
exit $fail
