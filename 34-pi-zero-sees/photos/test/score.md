# Hallucination test: 20 of my own photos, scored by hand (27 Nov 2026)

Photos: Chip (OV3660), VGA 640x480, my desk, one setup each; truth typed by me in the booth (`results.jsonl`).
Models on the Mac (CPU, -c 512, temp 0, Q8_0). The Pi gives the same words for the same photo and prompt
(checked on photo 1, photo 2 of the earlier set, and the batch2 photos below).

Rules: **R** = names the real object(s), nothing wrong said about them. **H** = half: too vague ("a device"),
misses an object in a multi-object shot, invents an extra thing, or breaks (a runaway list). **W** = wrong:
calls the main object something else, or no usable answer ("1. 1."). Lenient on things that really are in the
frame: the purple desk organiser at the left edge, the mouse's USB cable, the Minion's blue backpack.

| # | Truth | 256M sentence | 256M list | 500M sentence | 500M list |
|---|---|---|---|---|---|
| 01 | pen | R | H "Pencil, table, light" | R | R |
| 02 | stapler | H "a white color object on the floor" | W "telephone" | R | W "1. 1." |
| 03 | smart watch | R | R reads the time, 06:52 | R "with the number 0652" | R |
| 04 | glasses | R | R | R | R |
| 05 | Minion toy | **W "a toy of a minivan and a toy of a minivan"** | H Minion + "toolbox, telephone" | R "holding a blue and white suitcase" | R |
| 06 | computer mouse | R (+ USB cable, real) | R | R | R |
| 07 | microcontroller | H "a device" | H "Camera, buttons" | H "small electronic device" | H "1 device" |
| 08 | remote | R | H invents a second remote | R "with a small screen" | R |
| 09 | calculator | **W "a mobile phone"** | W "A cellphone" | R | H + "Phone, paper, ruler" |
| 10 | pen, calculator | R (+ "a pouch") | H "Phone, pen, cover" | R | H runaway: keyboard, paper ×9… |
| 11 | smart phone | R | R | H + "an iPod dock" | R |
| 12 | multimeter | H "a device" | H "Device, cord" | R "black and red multimeter" | W "1. 1." |
| 13 | USB hub | H "a white color object" | R "adapter" | H "a white USB cable" | H "white rectangular object" |
| 14 | earphones | R | R | R | H "3 black earphones" |
| 15 | glasses, Minion | W runaway "on the table there is a table…" | R "Glasses, mini plushies" | H "two pairs of glasses" | R |
| 16 | pen, stapler, calculator | H misses the stapler | H "A Casio" | H calculator only | H "phone, phone, phone…" ×10 |
| 17 | microcontroller, multimeter | H runaway "a device… a device…" | H "electronic components" | H "electronic devices" | H both as "devices" + a "diagram" |
| 18 | AA battery | **W "a lighter"** | W "Relish, lighter" | **W "a small metallic cigar lighter"** | W "Cigarette lighter" |
| 19 | ink bottle | R | R | R "small ink bottle" | R |
| 20 | box of staples | **W "a camera box"** | R "Staples, 240" (reads the box) | R | R |

## Totals

| | Right | Half | Wrong |
|---|---|---|---|
| 256M, one sentence | **9** | 6 | **5** |
| 256M, list | 9 | 8 | 3 |
| 500M, one sentence | **13** | 6 | **1** |
| 500M, list | 10 | 7 | 3 |

Headline (one-sentence prompt, the one the video uses): **the 256M gets 9 of 20 right and 5 flat wrong; the 500M
gets 13 right and 1 wrong.** The one the 500M gets wrong, the battery, the 256M gets wrong in the same way: both
see a lighter.

## For the script
- Best wrong answers: Minion → "a toy of a minivan, and a toy of a minivan" (256M); calculator → mobile phone;
  staples box → "a camera box", although the list prompt reads "Staples" off the same box; battery → "lighter" (both).
- Best right answers: the watch's time read correctly (06:52); the 500M naming the multimeter and the ink bottle.
- New failure type: the **list** prompt sometimes breaks the 500M ("1. 1.", "phone, phone, phone…"). The list
  prompt that rescued photo 2 of the earlier set isn't better overall (500M: 10 right vs 13 with one sentence).
- Neither model knows what a microcontroller is: "a device", every time. *(Chip's reaction shot.)*

## Pi spot checks (256M, one sentence, 27 Nov 19:13–19:19; logs in code/pi/logs/)
- 05 Minion: "a toy of a minivan and a toy of a minivan" — same as the Mac (111.4 s)
- 09 calculator: "a mobile phone on the table" — same (120.2 s)
- 18 battery: "a lighter" — same (108.7 s)
- 20 staples box: **"a box with some text on it"** — differs from the Mac's "a camera box" (129.5 s). Same files, temp 0;
  the Pi's CPU maths rounds differently (like the 500M's "3D earbuds"). Neither is "staples".
