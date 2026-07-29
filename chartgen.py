#!/usr/bin/env python3
"""
chartgen.py - 7kRose chart generator

Turns an audio file into easy/medium/hard note charts in the format charts.js
expects:  [ time , pitchRank , holdEnd ]

Every constant below was reverse-engineered from the seven charts that shipped
with the game, so generated songs sit alongside the originals rather than
feeling like a different game:

    grid    1024-sample hop @ 44100 Hz -> 0.02322 s; every original note is on it
    density hard 8.5 notes/sec; medium keeps 60.0% of that; easy keeps 32.5%
    lanes   dead even 25/25/25/25 at 4K, at every difficulty
    chords  4.2% / 10.2% / 17.4% of note pairs, staggered one frame apart -
            the originals contain zero truly simultaneous notes
    holds   ~21% / ~12% / ~5%, min length 0.24 s, hard cap 1.5 s

Usage:
    python3 chartgen.py song8.mp3 --title "Track Name" --id song8
    python3 chartgen.py *.mp3 --out charts_new.js
    python3 chartgen.py song8.mp3 --append charts.js

Requires: ffmpeg, ffprobe, numpy
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

SR = 44100
HOP = 1024
WIN = 2048
FRAME_DT = HOP / SR  # 0.023219...

N_BANDS = 8    # frequency bands analysed independently, which is what allows chords
MAX_CHORD = 4  # most notes allowed in one chord cluster
MIN_SEP = 2    # frames between non-chord notes, so adjacency stays budgeted

# nps   = notes per second (hard); keep = fraction of that budget
# chord = fraction of notes that are chord partners. Without an explicit budget,
#         per-band detection yields ~51% chords at every difficulty, because one
#         drum hit lights up every band at once.
DIFFS = {
    "hard":   {"nps": 8.5, "keep": 1.000, "chord": 0.205},
    "medium": {"nps": 8.5, "keep": 0.600, "chord": 0.120},
    "easy":   {"nps": 8.5, "keep": 0.325, "chord": 0.050},
}

HOLD_MIN_GAP = 0.26        # s of empty space needed before a hold is considered
HOLD_TAIL_MARGIN = 0.09    # s of breathing room left before the next note
HOLD_MIN_LEN = 0.235       # s - anything shorter is just a tap
HOLD_MAX_LEN = 1.5         # s - the original charts cap every hold here
HOLD_SUSTAIN_RATIO = 0.55  # energy must stay this fraction of the onset energy


def decode(path):
    """Decode any audio file to mono float32 at SR via ffmpeg."""
    cmd = ["ffmpeg", "-v", "quiet", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def duration(path):
    """Exact duration in seconds from ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "csv=p=0", path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return round(float(out.decode().strip()), 2)


def framed(y):
    """Overlapping analysis frames on the original chart grid."""
    n = 1 + max(0, (len(y) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return y[idx]


def spectrogram(y):
    frames = framed(y) * np.hanning(WIN).astype(np.float32)
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)


def rms_envelope(y):
    """Per-frame RMS on the analysis grid, for sustain detection."""
    return np.sqrt((framed(y).astype(np.float32) ** 2).mean(axis=1))


def onset_candidates(spec):
    """
    Per-band spectral flux, peak-picked independently in each band.

    Summing flux across the whole spectrum into a single envelope can only ever
    produce one note per instant, which is why a first attempt scored 0% chords
    against the original's 17%. Detecting per band lets a kick and a hat on the
    same beat become two notes in two different lanes.

    Returns (frames, pitch_values, strengths).
    """
    log_spec = np.log1p(spec * 1000.0)
    flux = np.diff(log_spec, axis=0)
    np.maximum(flux, 0, out=flux)
    flux = np.vstack([np.zeros((1, flux.shape[1]), np.float32), flux])

    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    cuts = np.geomspace(40.0, min(16000.0, freqs[-1]), N_BANDS + 1)
    edges = np.searchsorted(freqs, cuts).clip(1, spec.shape[1])

    k = 21
    out_f, out_v, out_s = [], [], []
    for b in range(N_BANDS):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        env = flux[:, lo:hi].sum(axis=1)
        pad = np.pad(env, (k // 2, k // 2), mode="edge")
        local = np.lib.stride_tricks.sliding_window_view(pad, k).mean(axis=1)
        env = np.maximum(env - local * 0.55, 0.0)
        # normalise per band, or the bass band wins everything and the top of
        # the keyboard never receives a note
        scale = env.std()
        if scale < 1e-9:
            continue
        env = env / scale

        prev = np.concatenate([[0.0], env[:-1]])
        nxt = np.concatenate([env[1:], [0.0]])
        pk = np.flatnonzero((env > prev) & (env >= nxt) & (env > 0))
        if len(pk) == 0:
            continue

        # pitch value = centroid within this band, so sorting by it orders by
        # band and still resolves pitch inside a band
        sub = spec[pk, lo:hi]
        e = sub.sum(axis=1)
        e[e == 0] = 1e-9
        cent = (sub * freqs[None, lo:hi]).sum(axis=1) / e

        out_f.append(pk)
        out_v.append(np.log2(np.maximum(cent, 20.0)))
        out_s.append(env[pk])

    if not out_f:
        return np.array([], int), np.array([]), np.array([])
    return np.concatenate(out_f), np.concatenate(out_v), np.concatenate(out_s)


def select(frames, values, strength, n_total, chord_frac):
    """
    Choose about n_total notes, of which chord_frac are chord partners.
    Returns a sorted list of (frame, pitch_value).

    Two structural rules, both taken from the shipped charts:

    * Non-chord notes sit at least MIN_SEP frames apart. Per-band detection
      otherwise scatters one-frame-apart pairs across the whole chart (10% of
      pairs on easy, where the original has 4% in total), because a kick and a
      hi-hat land on neighbouring frames and each becomes its own note.

    * Chord partners are staggered one frame late rather than sharing a
      timestamp. The originals contain zero truly simultaneous notes at any
      difficulty - every chord is exactly one frame (23 ms) apart, which is
      imperceptible to play but never stacks two notes on one instant.
    """
    order = np.argsort(-strength)
    best_at, extras = {}, []
    for i in order:
        f = int(frames[i])
        if f in best_at:
            extras.append(i)
        else:
            best_at[f] = i

    n_chord = int(round(n_total * chord_frac))
    n_prim = max(1, n_total - n_chord)

    taken = set()
    primaries = []
    for i in sorted(best_at.values(), key=lambda j: -strength[j]):
        if len(primaries) >= n_prim:
            break
        f = int(frames[i])
        if any((f + d) in taken for d in range(-MIN_SEP, MIN_SEP + 1)):
            continue
        taken.add(f)
        primaries.append(i)

    notes = [(int(frames[i]), float(values[i])) for i in primaries]

    live = {int(frames[i]) for i in primaries}
    per_frame, picked = {}, 0
    for i in extras:
        if picked >= n_chord:
            break
        f = int(frames[i])
        if f not in live or per_frame.get(f, 1) >= MAX_CHORD:
            continue
        slot = f + per_frame.get(f, 1)
        if slot in taken:
            continue
        taken.add(slot)
        per_frame[f] = per_frame.get(f, 1) + 1
        notes.append((slot, float(values[i])))
        picked += 1

    notes.sort()
    return notes


def to_ranks(values):
    """
    Percentile-rank pitch values into pitchRank 0..999.

    Ranking rather than mapping raw frequency is what yields the dead-even lane
    split the shipped charts have. It must be recomputed per difficulty over
    only that difficulty's notes: loudness correlates with pitch, so ranking
    across all onsets and then keeping the loudest pushes an easy chart onto one
    side of the keyboard.
    """
    n = len(values)
    if n <= 1:
        return np.array([500] * n)
    order = np.argsort(np.argsort(values))
    return np.round(order / (n - 1) * 999).astype(int)


def add_holds(pairs, rms):
    """
    Decide which notes become holds, per difficulty. This is why easy charts end
    up with more holds than hard ones: the sparser the chart, the more empty
    space a note has to sustain into.
    """
    out = []
    for i, (frame, rank) in enumerate(pairs):
        t = frame * FRAME_DT
        end = 0.0
        if i + 1 < len(pairs):
            gap = (pairs[i + 1][0] - frame) * FRAME_DT
            if gap >= HOLD_MIN_GAP:
                floor = (rms[frame] if frame < len(rms) else 0.0) * HOLD_SUSTAIN_RATIO
                limit = pairs[i + 1][0]
                f = frame + 1
                while f < limit and f < len(rms) and rms[f] >= floor:
                    f += 1
                candidate = min((f - frame) * FRAME_DT,
                                gap - HOLD_TAIL_MARGIN,
                                HOLD_MAX_LEN)
                if candidate >= HOLD_MIN_LEN:
                    end = round(t + candidate, 3)
        out.append([round(t, 3), int(rank), end])
    return out


def split_ties(notes):
    """
    Separate only chord notes whose pitchRank is genuinely identical. Chord
    partners come from different bands so their ranks already differ; all this
    fixes is an exact collision, which would otherwise be dropped by the engine
    for stacking on an occupied lane.
    """
    i = 0
    while i < len(notes):
        j = i
        while j + 1 < len(notes) and notes[j + 1][0] == notes[i][0]:
            j += 1
        if j > i:
            seen = set()
            for k in range(i, j + 1):
                r = notes[k][1]
                while r in seen:
                    r = (r + 250) % 1000
                seen.add(r)
                notes[k][1] = r
        i = j + 1
    return notes


def generate(path, title=None, sid=None):
    name = os.path.splitext(os.path.basename(path))[0]
    sid = sid or re.sub(r"\W+", "", name) or "song"
    title = title or name.replace("_", " ").replace("-", " ").strip()

    y = decode(path)
    dur = duration(path)
    spec = spectrogram(y)
    rms = rms_envelope(y)
    frames, values, strength = onset_candidates(spec)
    if len(frames) == 0:
        raise RuntimeError("no onsets found in " + path)

    charts = {}
    n_hard = int(round(DIFFS["hard"]["nps"] * dur))
    for diff in ("easy", "medium", "hard"):
        cfg = DIFFS[diff]
        n = max(1, int(round(n_hard * cfg["keep"])))
        picked = select(frames, values, strength, n, cfg["chord"])
        ranks = to_ranks(np.array([v for _, v in picked]))
        pairs = [(f, int(r)) for (f, _), r in zip(picked, ranks)]
        charts[diff] = split_ties(add_holds(pairs, rms))

    return {"id": sid, "file": os.path.basename(path), "title": title,
            "dur": dur, "charts": charts}


def as_js(song, per_line=8):
    """Render one song entry in the same shape charts.js already uses."""
    lines = ["  {",
             '    id:"%s", file:"%s", title:%s, dur:%g,'
             % (song["id"], song["file"], json.dumps(song["title"]), song["dur"]),
             "    charts:{"]
    for di, diff in enumerate(("easy", "medium", "hard")):
        notes = song["charts"][diff]
        chunks = []
        for i in range(0, len(notes), per_line):
            chunks.append("        " + ",".join(
                "[%g,%d,%g]" % (n[0], n[1], n[2]) for n in notes[i:i + per_line]))
        comma = "," if di < 2 else ""
        lines.append("      %s:[\n%s\n      ]%s" % (diff, ",\n".join(chunks), comma))
    lines += ["    }", "  }"]
    return "\n".join(lines)


def report(song):
    print("\n%s  (%gs)" % (song["title"], song["dur"]))
    print("  %-8s %6s %6s %8s %7s %s"
          % ("diff", "notes", "nps", "chords", "holds", "lanes 4K"))
    for diff in ("easy", "medium", "hard"):
        notes = song["charts"][diff]
        span = notes[-1][0] - notes[0][0] if len(notes) > 1 else 1
        holds = sum(1 for n in notes if n[2] > 0)
        chords = sum(1 for i in range(len(notes) - 1)
                     if notes[i + 1][0] - notes[i][0] < 0.024)
        lanes = [0, 0, 0, 0]
        for n in notes:
            lanes[min(3, int(n[1] / 1000 * 4))] += 1
        spread = " ".join("%.0f%%" % (100.0 * l / len(notes)) for l in lanes)
        print("  %-8s %6d %6.2f %7.1f%% %6.1f%% %s"
              % (diff, len(notes), len(notes) / span,
                 100.0 * chords / max(1, len(notes) - 1),
                 100.0 * holds / len(notes), spread))


def main():
    ap = argparse.ArgumentParser(description="Generate 7kRose charts from audio.")
    ap.add_argument("files", nargs="+", help="audio files (mp3, wav, ogg, ...)")
    ap.add_argument("--title", help="song title (single file only)")
    ap.add_argument("--id", help="song id (single file only)")
    ap.add_argument("--out", help="write the JS block to this file")
    ap.add_argument("--append", metavar="CHARTS_JS",
                    help="insert into an existing charts.js before the closing ];")
    args = ap.parse_args()

    if len(args.files) > 1 and (args.title or args.id):
        ap.error("--title/--id only work with a single file")

    songs = []
    for path in args.files:
        if not os.path.exists(path):
            print("skipping missing file: " + path, file=sys.stderr)
            continue
        print("analyzing %s ..." % os.path.basename(path), file=sys.stderr)
        song = generate(path, args.title, args.id)
        report(song)
        songs.append(song)

    if not songs:
        sys.exit("nothing generated")

    blocks = ",\n".join(as_js(s) for s in songs)

    if args.append:
        src = open(args.append, encoding="utf-8").read()
        m = re.search(r"\n\];\s*$", src)
        if not m:
            sys.exit("could not find the closing '];' in " + args.append)
        backup = args.append + ".bak"
        if not os.path.exists(backup):
            open(backup, "w", encoding="utf-8").write(src)
            print("\nbacked up %s -> %s" % (args.append, os.path.basename(backup)),
                  file=sys.stderr)
        open(args.append, "w", encoding="utf-8").write(
            src[:m.start()] + ",\n" + blocks + "\n];\n")
        print("appended %d song(s) to %s" % (len(songs), args.append), file=sys.stderr)
    elif args.out:
        open(args.out, "w", encoding="utf-8").write(blocks + "\n")
        print("\nwrote " + args.out, file=sys.stderr)
    else:
        print(blocks)


if __name__ == "__main__":
    main()
