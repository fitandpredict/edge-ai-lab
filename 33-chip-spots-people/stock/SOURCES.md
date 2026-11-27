# Stock footage used as Chip's test scenes (episode 33)

All clips are from Pexels under the Pexels License (free to use and modify, commercial use OK, attribution
not required; we credit Pexels in the description anyway). Downloaded 27 Nov 2026. Chip never saw them through
its lens: `tools/feed_clips.py` sends the frames over USB (inject mode), centre-cropped to 4:3 and scaled to
640x480 like Chip's VGA camera. Reference labels: YOLO11x on the Mac (`tools/make_testset.py`), spot-checked by eye.

| Clip | Pexels page | What it tests |
|---|---|---|
| 5311414 | https://www.pexels.com/video/a-man-using-laptop-while-in-the-office-5311414/ | person at a desk |
| 4251333 | https://www.pexels.com/video/video-of-a-man-using-a-laptop-4251333/ | person at a desk, side view |
| 8512941 | https://www.pexels.com/video/a-woman-looking-on-her-laptop-8512941/ | person at a desk |
| 7614413 | https://www.pexels.com/video/man-using-his-cellphone-while-walking-at-home-7614413/ | walks through, room goes empty |
| 7694274 | https://www.pexels.com/video/a-blind-man-walking-in-the-living-room-7694274/ | walking, bending, sitting |
| 5977455 | https://www.pexels.com/video/colleagues-discussing-a-documents-while-walking-inside-the-office-5977455/ | two people, far to very close |
| 5971592 | https://www.pexels.com/video/a-woman-walking-inside-the-workplace-5971592/ | small, far-away person |
| 7685810 | https://www.pexels.com/video/a-man-going-out-to-the-exit-door-7685810/ | walks away, behind glass |
| 7597582 | https://www.pexels.com/video/an-anxious-person-indoors-7597582/ | legs only |
| 8347237 | https://www.pexels.com/video/an-empty-office-8347237/ | nobody (false alarms) |
| 6951393 | https://www.pexels.com/video/an-empty-conference-room-6951393/ | nobody, moving camera |
| 4139630 | https://www.pexels.com/video/dogs-barking-at-the-tv-4139630/ | dogs and a TV, nobody |

Test set `test60.txt`: 5 evenly spaced frames per clip = 60 frames, 36 with a person, 24 without.
