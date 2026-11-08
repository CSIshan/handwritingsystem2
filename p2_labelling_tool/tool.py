"""
P2 Labelling & Verification Tool
==================================
Terminal UI using curses. Zero external dependencies.

Usage:
    from handwriting_system.p2_labelling_tool import LabellingTool
    tool = LabellingTool()
    tool.run()

Keys:
    A — Accept
    E — Edit label
    D — Discard
    S — Skip (revisit later)
    R — Replay stroke animation
    Q — Quit and save
"""
import curses
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from .loader   import load_raw_dataset
from .progress import Progress
from .renderer import render_image_ascii, render_strokes_frame, animate_strokes
from .saver    import save_verified

log = logging.getLogger(__name__)


class LabellingTool:

    def __init__(
        self,
        raw_dir:          str   = "dataset/raw",
        verified_dir:     str   = "dataset/verified",
        progress_path:    str   = "dataset/progress.json",
        spot_check_ratio: float = 0.15,
    ):
        self.raw_dir          = raw_dir
        self.verified_dir     = verified_dir
        self.progress_path    = progress_path
        self.spot_check_ratio = spot_check_ratio

    def run(self):
        """Launch the curses UI."""
        try:
            records  = load_raw_dataset(self.raw_dir, self.spot_check_ratio)
            progress = Progress(self.progress_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Filter already-decided records
        pending = [r for r in records if not progress.is_decided(r["_record_id"])]

        if not pending:
            print("All records already reviewed. Nothing to do.")
            s = progress.summary()
            print(f"Summary: {s['accepted']} accepted, {s['edited']} edited, "
                  f"{s['discarded']} discarded")
            save_verified(records, progress, self.verified_dir)
            return

        try:
            curses.wrapper(self._ui_main, records, pending, progress)
        except KeyboardInterrupt:
            pass
        finally:
            save_verified(records, progress, self.verified_dir)
            s = progress.summary()
            print(f"\nSession complete:")
            print(f"  Accepted:  {s['accepted']}")
            print(f"  Edited:    {s['edited']}")
            print(f"  Discarded: {s['discarded']}")
            print(f"  Total:     {s['total']}")
            print(f"  Verified saved to: {self.verified_dir}")

    def _ui_main(self, stdscr, all_records, pending, progress):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()

        # Colour pairs
        curses.init_pair(1, curses.COLOR_GREEN,  -1)   # accepted
        curses.init_pair(2, curses.COLOR_RED,    -1)   # flagged / discard
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # edit / warning
        curses.init_pair(4, curses.COLOR_CYAN,   -1)   # info
        curses.init_pair(5, curses.COLOR_WHITE,  -1)   # normal

        GREEN  = curses.color_pair(1)
        RED    = curses.color_pair(2)
        YELLOW = curses.color_pair(3)
        CYAN   = curses.color_pair(4)

        idx = 0
        skipped = []

        while idx < len(pending):
            record = pending[idx]
            rid    = record["_record_id"]

            # Skip already-decided (could have been decided in this session)
            if progress.is_decided(rid):
                idx += 1
                continue

            stdscr.clear()
            h, w = stdscr.getmaxyx()

            # ── Header ───────────────────────────────────────────────────────
            remaining = len(pending) - idx
            header = (
                f" P2 LABELLING TOOL  "
                f"[{remaining} remaining | "
                f"{progress.summary()['accepted']} accepted | "
                f"{progress.summary()['discarded']} discarded] "
            )
            stdscr.addstr(0, 0, header[:w-1], CYAN | curses.A_BOLD)
            stdscr.addstr(1, 0, "─" * (w - 1), CYAN)

            # ── Source info ───────────────────────────────────────────────────
            source    = record.get("_source_file", "unknown")
            pair_idx  = record.get("source", {}).get("pair_index", "?")
            word_idx  = record.get("source", {}).get("word_index", "?")
            is_flagged= record.get("flagged", False)
            flag_why  = record.get("flag_reason", "")
            is_spot   = record.get("_spot_check", False)

            info = f" Source: {source}  Pair: {pair_idx}  Word: {word_idx}"
            stdscr.addstr(2, 0, info[:w-1])

            if is_flagged:
                flag_str = f" ⚑ FLAGGED: {flag_why}"
                stdscr.addstr(3, 0, flag_str[:w-1], RED | curses.A_BOLD)
            elif is_spot:
                stdscr.addstr(3, 0, " ✓ SPOT CHECK (clean word sample)", YELLOW)
            else:
                stdscr.addstr(3, 0, " ✓ CLEAN", GREEN)

            stdscr.addstr(4, 0, "─" * (w - 1))

            # ── Stroke render ─────────────────────────────────────────────────
            strokes     = record.get("strokes", [])
            render_w    = min(50, w - 4)
            render_h    = 8
            stroke_lines= render_strokes_frame(strokes, render_w, render_h)

            stdscr.addstr(5, 0, " STROKE PATH:", CYAN)
            for i, line in enumerate(stroke_lines):
                stdscr.addstr(6 + i, 2, line[:w-3])

            # ── Label ─────────────────────────────────────────────────────────
            label_y = 6 + render_h + 1
            label   = record.get("word", "")
            bbox    = record.get("bbox_mm", {})
            bbox_str= (f"  w={bbox.get('width',0):.1f}mm "
                       f"asc={bbox.get('ascent',0):.1f}mm "
                       f"desc={bbox.get('descent',0):.1f}mm")

            stdscr.addstr(label_y,     0, f" Label: ", CYAN)
            stdscr.addstr(label_y,     8, label, curses.A_BOLD)
            stdscr.addstr(label_y + 1, 0, bbox_str)
            stdscr.addstr(label_y + 2, 0,
                          f" Points: {len(strokes)}  "
                          f"Strokes: {sum(1 for s in strokes if s[2]==1)}")

            # ── Controls ──────────────────────────────────────────────────────
            ctrl_y = label_y + 4
            stdscr.addstr(ctrl_y,     0, "─" * (w - 1))
            stdscr.addstr(ctrl_y + 1, 0,
                " [A] Accept  [E] Edit label  [D] Discard  [S] Skip  "
                "[R] Replay  [Q] Quit")

            stdscr.refresh()

            # ── Wait for keypress ─────────────────────────────────────────────
            while True:
                key = stdscr.getch()
                ch  = chr(key).upper() if 0 < key < 256 else ""

                if ch == "A":
                    progress.accept(rid)
                    idx += 1
                    break

                elif ch == "D":
                    progress.discard(rid)
                    idx += 1
                    break

                elif ch == "S":
                    skipped.append(record)
                    idx += 1
                    break

                elif ch == "Q":
                    return   # finally block handles save

                elif ch == "R":
                    # Replay animation inline
                    def draw_frame(lines):
                        for i, line in enumerate(lines):
                            stdscr.addstr(6 + i, 2, " " * render_w)
                            stdscr.addstr(6 + i, 2, line[:w-3])
                        stdscr.refresh()

                    curses.curs_set(0)
                    animate_strokes(strokes, draw_frame,
                                    width=render_w, height=render_h)

                elif ch == "E":
                    # Edit label — show input field
                    edit_y = label_y
                    stdscr.addstr(edit_y, 0, " Label: [              ]")
                    stdscr.addstr(edit_y, 9, " " * 14)
                    stdscr.move(edit_y, 9)
                    curses.curs_set(1)
                    curses.echo()

                    new_label = ""
                    try:
                        raw = stdscr.getstr(edit_y, 9, 20)
                        new_label = raw.decode("utf-8").strip()
                    except Exception:
                        pass
                    finally:
                        curses.noecho()
                        curses.curs_set(0)

                    if new_label:
                        progress.edit(rid, new_label, label)
                        idx += 1
                        break
                    else:
                        # Empty input — redraw label and continue
                        stdscr.addstr(edit_y, 0, f" Label: ")
                        stdscr.addstr(edit_y, 8, label + "          ",
                                      curses.A_BOLD)
                        stdscr.refresh()

        # Append skipped to end for second pass
        if skipped:
            pending.extend(skipped)
            skipped = []
