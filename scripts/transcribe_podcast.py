#!/usr/bin/env python3
"""
小宇宙播客转录工具
从 小宇宙 FM 获取播客音频，使用 Qwen3-ASR 本地转录为文字。

用法:
  python3 transcribe_podcast.py --token TOKEN --keyword "关键词"
  python3 transcribe_podcast.py --token TOKEN --eid EPISODE_ID
  python3 transcribe_podcast.py --token TOKEN --eid EPISODE_ID --model-dir /path/to/model --asr-bin /path/to/demo
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

BASE_URL = os.environ.get("XYZ_BASE_URL", "http://localhost:23020")
AUDIO_DIR = Path(tempfile.gettempdir()) / "xiaoyuzhou-audio"
MAX_SEGMENT_SEC = 180  # 3 min — Metal GPU hangs on longer segments


def api(endpoint: str, token: str, payload: dict, _retry: bool = True) -> dict:
    """Call xyz API endpoint with auto token refresh on 401."""
    import urllib.request
    import urllib.error
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-jike-access-token": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 401 and _retry:
            refresh_token = os.environ.get("XYZ_REFRESH_TOKEN")
            if refresh_token:
                print("  Token 过期，自动刷新...")
                refresh_req = urllib.request.Request(
                    f"{BASE_URL}/refresh_token",
                    data=json.dumps({"refreshToken": refresh_token}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(refresh_req) as resp:
                    result = json.loads(resp.read())
                new_token = result.get("data", {}).get("token", "")
                if new_token:
                    os.environ["XYZ_ACCESS_TOKEN"] = new_token
                    return api(endpoint, new_token, payload, _retry=False)
        raise


def search_episodes(token: str, keyword: str, limit: int = 5) -> list:
    """Search episodes by keyword."""
    result = api("/search", token, {"keyword": keyword, "type": "EPISODE"})
    episodes = []
    for item in result.get("data", {}).get("data", []):
        if item.get("type") == "EPISODE":
            episodes.append(item)
    return episodes[:limit]


def get_episode_detail(token: str, eid: str) -> dict:
    """Get episode detail by eid."""
    result = api("/episode_detail", token, {"eid": eid})
    return result.get("data", {}).get("data", {})


def download_audio(url: str, output_path: Path) -> Path:
    """Download audio file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  下载音频: {url[:80]}...")
    urlretrieve(url, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  已下载: {size_mb:.1f} MB")
    return output_path


def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert audio to WAV 16kHz mono using ffmpeg."""
    print("  转换为 WAV 16kHz...")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-f", "wav", str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg 错误: {result.stderr[:200]}", file=sys.stderr)
        sys.exit(1)
    return output_path


def get_duration_sec(wav_path: Path) -> float:
    """Get WAV duration in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def split_audio(wav_path: Path, output_dir: Path) -> list[Path]:
    """Split audio at silence boundaries if > MAX_SEGMENT_SEC."""
    duration = get_duration_sec(wav_path)
    if duration <= MAX_SEGMENT_SEC:
        print(f"  音频时长: {duration:.0f}s ({duration/60:.1f}min), 无需分割")
        return [wav_path]

    print(f"  音频时长: {duration:.0f}s ({duration/60:.1f}min), 需要分割")

    # Detect silence points
    result = subprocess.run(
        ["ffmpeg", "-i", str(wav_path),
         "-af", "silencedetect=noise=-30dB:d=2",
         "-f", "null", "-"],
        capture_output=True, text=True
    )

    # Parse silence end times
    ends = re.findall(r"silence_end:\s*([\d.]+)", result.stderr)
    MIN_SEGMENT_SEC = 60
    split_times = []
    for t in ends:
        sec = float(t)
        if sec < MIN_SEGMENT_SEC:
            continue
        # Only split at boundaries roughly every MAX_SEGMENT_SEC
        if not split_times or sec - split_times[-1] >= MAX_SEGMENT_SEC * 0.8:
            split_times.append(sec)

    # Ensure we don't miss the tail
    if not split_times or duration - split_times[-1] > MAX_SEGMENT_SEC:
        # Add evenly spaced splits
        t = MAX_SEGMENT_SEC
        while t < duration:
            if not any(abs(t - s) < 60 for s in split_times):
                split_times.append(t)
            t += MAX_SEGMENT_SEC

    split_times.sort()
    if not split_times:
        split_times = [MAX_SEGMENT_SEC]

    print(f"  分割点: {[f'{t:.0f}s' for t in split_times]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-f", "segment",
        "-segment_times", ",".join(f"{t:.3f}" for t in split_times),
        "-ar", "16000", "-ac", "1",
        str(output_dir / "seg_%03d.wav")
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    segments = sorted(output_dir.glob("seg_*.wav"))
    print(f"  分割为 {len(segments)} 个片段")
    return segments


def ensure_tokenizer(model_dir: str) -> None:
    """Build tokenizer.json from vocab.json + merges.txt if missing."""
    tok_path = Path(model_dir) / "tokenizer.json"
    if tok_path.exists():
        return
    print("  生成 tokenizer.json...")
    vocab_path = Path(model_dir) / "vocab.json"
    merges_path = Path(model_dir) / "merges.txt"
    config_path = Path(model_dir) / "tokenizer_config.json"
    if not all(p.exists() for p in [vocab_path, merges_path, config_path]):
        print("  缺少 vocab.json / merges.txt / tokenizer_config.json", file=sys.stderr)
        sys.exit(1)

    with open(vocab_path) as f:
        vocab_val = json.load(f)
    with open(merges_path) as f:
        merges_vec = [l for l in f.read().splitlines() if l and not l.startswith('#')]
    with open(config_path) as f:
        tok_cfg = json.load(f)

    added_tokens = []
    if "added_tokens_decoder" in tok_cfg:
        entries = sorted(
            [(int(k), v) for k, v in tok_cfg["added_tokens_decoder"].items()],
            key=lambda x: x[0]
        )
        for id_, v in entries:
            added_tokens.append({
                "id": id_, "content": v["content"],
                "single_word": False, "lstrip": False, "rstrip": False,
                "normalized": False, "special": v.get("special", False)
            })

    tokenizer_json = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab_val, "merges": merges_vec}
    }
    with open(tok_path, "w") as f:
        json.dump(tokenizer_json, f)
    print("  tokenizer.json 已生成")


def transcribe_segment(wav_path: Path, model_dir: str, asr_bin: str) -> str:
    """Transcribe a single WAV segment using qwen3-asr-rs local_transcribe binary."""
    cmd = [asr_bin, model_dir, str(wav_path)]
    print(f"  转录: {wav_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  转录错误: {result.stderr[:200]}", file=sys.stderr)
        return ""
    # Parse output: last line starting with "Text     :"
    lines = result.stdout.strip().splitlines()
    for line in lines:
        if line.startswith("Text     :"):
            return line[len("Text     :"):].strip()
    return result.stdout.strip()


def transcribe_segments(segments: list[Path], model_dir: str, asr_bin: str) -> str:
    """Transcribe multiple segments and combine."""
    texts = []
    for i, seg in enumerate(segments):
        print(f"  [{i+1}/{len(segments)}]", end=" ")
        text = transcribe_segment(seg, model_dir, asr_bin)
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def format_output(episode: dict, transcript: str) -> str:
    """Format transcript as markdown."""
    title = episode.get("title", "未知标题")
    podcast = episode.get("podcast", {})
    podcast_title = podcast.get("title", "未知节目")
    pub_date = episode.get("pubDate", "")
    duration = episode.get("duration", 0)
    mins, secs = divmod(duration, 60)
    play_count = episode.get("playCount", 0)

    lines = [
        f"# {title}",
        "",
        f"**节目**: {podcast_title}",
        f"**日期**: {pub_date[:10] if pub_date else '未知'}",
        f"**时长**: {int(mins)}分{int(secs)}秒",
    ]
    if play_count:
        lines.append(f"**播放量**: {play_count:,}")

    lines += [
        "",
        "---",
        "",
        "## 转录文本",
        "",
        transcript,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="小宇宙播客转录工具")
    parser.add_argument("--token", required=True, help="x-jike-access-token")
    parser.add_argument("--keyword", help="搜索关键词")
    parser.add_argument("--eid", help="单集 ID (可替代关键词搜索)")
    parser.add_argument("--model-dir", default="/Users/pite/qwen3-asr-models/0.6B",
                        help="Qwen3-ASR 模型目录")
    parser.add_argument("--asr-bin",
                        default="/Users/pite/qwen3-asr-rs/target/release/examples/local_transcribe",
                        help="qwen3-asr-rs local_transcribe 二进制路径")
    parser.add_argument("--output", "-o", help="输出文件路径 (默认 stdout)")
    parser.add_argument("--keep-audio", action="store_true", help="保留下载的音频文件")
    args = parser.parse_args()

    if not args.eid and not args.keyword:
        parser.error("需要 --eid 或 --keyword")

    # Step 1: Find episode
    if args.eid:
        print(f"获取单集详情: {args.eid}")
        eid = args.eid
        episode = get_episode_detail(args.token, eid)
    else:
        print(f"搜索: {args.keyword}")
        episodes = search_episodes(args.token, args.keyword)
        if not episodes:
            print("未找到相关单集", file=sys.stderr)
            sys.exit(1)
        print(f"找到 {len(episodes)} 个单集，选择第一个:")
        print(f"  {episodes[0].get('title', '?')}")
        eid = episodes[0]["eid"]
        episode = get_episode_detail(args.token, eid)

    title = episode.get("title", "未知")
    print(f"\n单集: {title}")

    # Step 2: Get audio URL
    media = episode.get("media", {})
    audio_url = media.get("source", {}).get("url") or episode.get("enclosure", {}).get("url")
    if not audio_url:
        print("未找到音频链接", file=sys.stderr)
        sys.exit(1)

    size_mb = media.get("size", 0) / 1024 / 1024
    print(f"音频: {size_mb:.1f} MB, {media.get('mimeType', 'unknown')}")

    # Step 3: Download and convert
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = eid or re.sub(r'[^\w\s-]', '', title)[:30]
    m4a_path = AUDIO_DIR / f"{safe_title}.m4a"
    wav_path = AUDIO_DIR / f"{safe_title}.wav"
    seg_dir = AUDIO_DIR / f"{safe_title}_segments"

    download_audio(audio_url, m4a_path)
    convert_to_wav(m4a_path, wav_path)

    # Step 4: Split if needed
    segments = split_audio(wav_path, seg_dir)

    # Step 5: Transcribe
    if not os.path.isfile(args.asr_bin):
        print(f"ASR 二进制不存在: {args.asr_bin}", file=sys.stderr)
        print("请先编译: cd qwen3-asr-rs && cargo build --release --example local_transcribe", file=sys.stderr)
        sys.exit(1)

    ensure_tokenizer(args.model_dir)

    print("\n开始转录...")
    transcript = transcribe_segments(segments, args.model_dir, args.asr_bin)

    # Step 6: Format output
    output = format_output(episode, transcript)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\n已保存到: {args.output}")
    else:
        print("\n" + output)

    # Cleanup
    if not args.keep_audio:
        for p in [m4a_path, wav_path]:
            p.unlink(missing_ok=True)
        if seg_dir.exists():
            for f in seg_dir.glob("*"):
                f.unlink()
            seg_dir.rmdir()

    print("\n完成!")


if __name__ == "__main__":
    main()
