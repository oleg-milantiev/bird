#!/usr/bin/env python3
"""
bird.resolve.py — generate a DaVinci Resolve-compatible FCPXML timeline
from a CSV of bird-detection frames.

Usage:
    python bird.resolve.py detections.csv

CSV:
    frame_number,video_path
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable


MIN_FRAMES = 10
SERIES_GAP = 60
PRE_ROLL = 2.0
POST_ROLL = 2.0
FCPXML_VERSION = "1.9"
OUTPUT_FILE = "timeline.fcpxml"


@dataclass
class Detection:
    frame: int
    path: str


@dataclass
class Series:
    path: str
    first_frame: int
    last_frame: int
    detection_count: int


@dataclass
class MediaInfo:
    path: str
    fps: Fraction
    duration: Fraction | None
    width: int | None
    height: int | None


def seconds_to_fraction(seconds: float) -> Fraction:
    return Fraction(str(seconds)).limit_denominator(1_000_000)


def fcpx_time(value: Fraction) -> str:
    if value.denominator == 1:
        return f"{value.numerator}s"
    return f"{value.numerator}/{value.denominator}s"


def frames_to_time(frames: int, fps: Fraction) -> Fraction:
    return Fraction(frames, 1) / fps


def run_ffprobe(path: str) -> dict:
    cmd = [
        r"E:\PORTABLE\ffmpeg\bin\ffprobe.exe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,width,height,duration:"
        "format=duration",
        "-of", "json",
        path,
    ]

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe не найден в PATH. Установите FFmpeg и проверьте "
            "команду 'ffprobe' в консоли."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe не смог прочитать файл:\n{path}\n{exc.stderr.strip()}"
        ) from exc

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Некорректный JSON от ffprobe: {path}") from exc


def parse_frame_rate(value: str | None) -> Fraction:
    if not value or value in ("0/0", "0"):
        raise ValueError("ffprobe не вернул корректный frame rate")
    fps = Fraction(value)
    if fps <= 0:
        raise ValueError(f"Некорректный frame rate: {value}")
    return fps


def get_media_info(path: str) -> MediaInfo:
    data = run_ffprobe(path)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"В файле нет видеопотока: {path}")

    stream = streams[0]
    # avg_frame_rate is normally the useful rate for a frame-number based
    # detector. Fall back to r_frame_rate if necessary.
    fps = parse_frame_rate(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    )

    duration_value = stream.get("duration")
    if duration_value is None:
        duration_value = (data.get("format") or {}).get("duration")

    duration = None
    if duration_value not in (None, "N/A"):
        duration = Fraction(str(duration_value)).limit_denominator(1_000_000)

    def as_int(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return MediaInfo(
        path=path,
        fps=fps,
        duration=duration,
        width=as_int(stream.get("width")),
        height=as_int(stream.get("height")),
    )


def read_csv(path: str) -> list[Detection]:
    result: list[Detection] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)

        if not reader.fieldnames:
            raise RuntimeError("CSV не содержит заголовка.")

        required = {"frame_number", "video_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "В CSV отсутствуют поля: " + ", ".join(sorted(missing))
            )

        for line_no, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame_number"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Строка {line_no}: некорректный frame_number: "
                    f"{row.get('frame_number')!r}"
                ) from exc

            video_path = (row.get("video_path") or "").strip()
            if not video_path:
                raise RuntimeError(f"Строка {line_no}: пустой video_path.")
            if frame < 0:
                raise RuntimeError(
                    f"Строка {line_no}: отрицательный frame_number: {frame}"
                )

            result.append(Detection(frame=frame, path=video_path))

    return result


def build_series(detections: Iterable[Detection]) -> list[Series]:
    """
    Group by source file, then split whenever the gap is > SERIES_GAP.

    Duplicate frame numbers are collapsed. A qualifying series contains
    MORE than MIN_FRAMES unique detected frames.
    """
    grouped: dict[str, set[int]] = {}

    for detection in detections:
        grouped.setdefault(detection.path, set()).add(detection.frame)

    result: list[Series] = []

    for path, frame_set in grouped.items():
        frames = sorted(frame_set)
        if not frames:
            continue

        first = last = frames[0]
        count = 1

        for frame in frames[1:]:
            if frame - last <= SERIES_GAP:
                last = frame
                count += 1
            else:
                result.append(
                    Series(path, first, last, count)
                )
                first = last = frame
                count = 1

        result.append(Series(path, first, last, count))

    return sorted(
        (s for s in result if s.detection_count > MIN_FRAMES),
        key=lambda s: (os.path.normcase(s.path), s.first_frame),
    )


def make_clips(
    series: list[Series],
    media: dict[str, MediaInfo],
) -> list[tuple[Series, Fraction, Fraction]]:
    """
    Return source ranges (series, start, end), merging overlapping ranges
    created by PRE_ROLL/POST_ROLL for the same source file.
    """
    by_path: dict[str, list[tuple[Series, Fraction, Fraction]]] = {}

    pre = seconds_to_fraction(PRE_ROLL)
    post = seconds_to_fraction(POST_ROLL)

    for s in series:
        info = media[s.path]

        # Frame N is treated as starting at N / FPS.
        # The detected range therefore ends just after last_frame.
        start = frames_to_time(s.first_frame, info.fps) - pre
        end = frames_to_time(s.last_frame + 1, info.fps) + post

        start = max(start, Fraction(0))
        if info.duration is not None:
            end = min(end, info.duration)

        if end > start:
            by_path.setdefault(s.path, []).append((s, start, end))

    result = []

    for path, clips in by_path.items():
        clips.sort(key=lambda x: x[1])
        current = clips[0]

        for nxt in clips[1:]:
            if nxt[1] <= current[2]:
                # Preserve the first series metadata, just extend its range.
                current = (current[0], current[1], max(current[2], nxt[2]))
            else:
                result.append(current)
                current = nxt

        result.append(current)

    return sorted(
        result,
        key=lambda x: (os.path.normcase(x[0].path), x[1]),
    )


def xml_escape(value: str) -> str:
    return html.escape(value, quote=True)


def file_url(path: str) -> str:
    """
    Produce a file:// URL. Windows paths are supported even if the script
    is being generated/tested on another OS.
    """
    from urllib.parse import quote

    absolute = os.path.abspath(path)

    try:
        return Path(absolute).as_uri()
    except ValueError:
        normalized = absolute.replace("\\", "/")
        if len(normalized) >= 2 and normalized[1] == ":":
            normalized = "/" + normalized
        return "file://" + quote(normalized, safe="/:")


def make_fcpxml(
    clips: list[tuple[Series, Fraction, Fraction]],
    media: dict[str, MediaInfo],
) -> str:
    first_info = next(iter(media.values()))

    width = first_info.width or 1920
    height = first_info.height or 1080
    fps = first_info.fps

    asset_ids: dict[str, str] = {}
    assets_xml = []

    for index, (path, info) in enumerate(media.items(), start=1):
        asset_id = f"r{index}"
        asset_ids[path] = asset_id

        duration = fcpx_time(info.duration) if info.duration is not None else "0s"

        assets_xml.append(
            f'            <asset id="{asset_id}" '
            f'name="{xml_escape(os.path.basename(path))}" '
            f'uid="{xml_escape(path)}" '
            f'duration="{duration}" '
            f'hasVideo="1" hasAudio="1">'
            f'<media-rep kind="original-media" '
            f'src="{xml_escape(file_url(path))}"/>'
            f'</asset>'
        )

    clip_xml = []
    timeline_position = Fraction(0)

    for series, start, end in clips:
        duration = end - start

        clip_xml.append(
            f'                        <asset-clip '
            f'name="{xml_escape(os.path.basename(series.path))} '
            f'[{series.first_frame}-{series.last_frame}]" '
            f'ref="{asset_ids[series.path]}" '
            f'offset="{fcpx_time(timeline_position)}" '
            f'start="{fcpx_time(start)}" '
            f'duration="{fcpx_time(duration)}" '
            f'enabled="1"/>'
        )

        timeline_position += duration

    timeline_duration = fcpx_time(timeline_position)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="{FCPXML_VERSION}">
    <resources>
        <format id="r-format"
                name="FFVideoFormat{width}x{height}"
                frameDuration="{fcpx_time(Fraction(1, 1) / fps)}"
                width="{width}"
                height="{height}"
                colorSpace="1-1-1 (Rec. 709)"/>
{os.linesep.join(assets_xml)}
    </resources>
    <library>
        <event name="Bird detections">
            <project name="Bird detections">
                <sequence format="r-format"
                          duration="{timeline_duration}"
                          tcStart="0s"
                          tcFormat="NDF"
                          audioLayout="stereo"
                          audioRate="48k">
                    <spine>
{os.linesep.join(clip_xml)}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DaVinci Resolve FCPXML from bird detections."
    )
    parser.add_argument("csv_file", help="CSV with frame_number,video_path")
    args = parser.parse_args()

    try:
        detections = read_csv(args.csv_file)
        if not detections:
            raise RuntimeError("CSV не содержит записей.")

        series = build_series(detections)

        print(f"Обнаружений: {len(detections)}")
        print(f"Подходящих серий (> {MIN_FRAMES}): {len(series)}")

        if not series:
            print("Подходящих серий нет. XML не создаётся.")
            return 0

        media: dict[str, MediaInfo] = {}
        for path in dict.fromkeys(s.path for s in series):
            print(f"ffprobe: {path}")
            media[path] = get_media_info(path)
            print(
                f"  fps={media[path].fps}, "
                f"duration={media[path].duration}"
            )

        clips = make_clips(series, media)
        print(f"Фрагментов после roll/merge: {len(clips)}")

        xml = make_fcpxml(clips, media)

        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(xml)

        print(f"Готово: {os.path.abspath(OUTPUT_FILE)}")

        for s, start, end in clips:
            print(
                f"  {os.path.basename(s.path)}: "
                f"frames {s.first_frame}-{s.last_frame}, "
                f"{s.detection_count} detections, "
                f"source {float(start):.3f}-{float(end):.3f}s"
            )

        return 0

    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
