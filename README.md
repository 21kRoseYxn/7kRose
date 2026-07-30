# 7kRose

A browser-based rhythm game in the style of osu!mania. Four-key top-scroll, seven tracks, three difficulties each, hold notes, rebindable keys, four themes, and a practice mode that loops any section of a song.

**[Play it here](https://21kroseyxn.github.io/7kRose/)**

No install, no accounts, nothing to download. On desktop the default keys are `D` `F` `J` `K` — rebind them in Settings — and `Esc` pauses. On a phone, tap the lanes.

## Music credits

Every track is used with its artist's permission. Please go support them.

| Track | Artist | Source |
|---|---|---|
| Armageddon | Zodin | [Newgrounds](https://www.newgrounds.com/audio/listen/548713) |
| Grizzly (WIP) | Envy | [Newgrounds](https://www.newgrounds.com/audio/listen/467065) |
| Silent Hill (Dubstep) | Aydin-Jewelz123 | [Newgrounds](https://www.newgrounds.com/audio/listen/386900) |
| {Rose} | cornandbeans | [Newgrounds](https://www.newgrounds.com/audio/listen/65711) |
| Nostalgia (original mix) | Acid-Notation | [YouTube](https://www.youtube.com/watch?v=dtpzCffp_y8) |
| Turbo Strawberry | Acid-Notation | [YouTube](https://www.youtube.com/watch?v=5OB-PHV_3Zo) |
| Heaven (remix) | Envy | [Newgrounds](https://www.newgrounds.com/audio/listen/105753) |

If you're one of these artists and you'd like your track removed, open an issue and it comes down the same day.

## chartgen.py — the chart generator

Charts aren't placed by hand. `chartgen.py` analyses an audio file and generates easy, medium and hard charts from it:

```bash
python3 chartgen.py newsong.mp3 --id song13 --title "Artist — Title" --append charts.js
```

Needs `ffmpeg`, `ffprobe` and `numpy`. A five-song batch takes about ten seconds.

It works by spectral flux onset detection across eight independent frequency bands, which is what allows two notes to land on the same beat in different lanes. Lane assignment comes from percentile-ranking each onset's spectral centroid, recomputed per difficulty. Every constant in it was calibrated by measuring the game's original hand-baked charts — the 0.02322s analysis grid, the 8.5 notes/sec ceiling, the 32.5%/60%/100% difficulty subsetting, and the chord rate that climbs 4% → 10% → 17% from easy to hard.

## Chart format

`charts.js` holds one entry per song. Each note is three numbers:

```js
[ time, pitchRank, holdEnd ]
```

`time` is seconds from the start of the audio. `pitchRank` is 0–999 and decides the lane — with `K` keys, `lane = floor(pitchRank / 1000 * K)`, so the same chart plays in 2K, 4K or 6K. `holdEnd` is `0` for a tap, or the second the hold releases.

## Running locally

Clone it and serve the folder over HTTP — opening `index.html` straight from disk will fail on audio decoding in most browsers.

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Credits

Built by Josh. An osu! fangame — not affiliated with or endorsed by osu! or ppy Pty Ltd.
