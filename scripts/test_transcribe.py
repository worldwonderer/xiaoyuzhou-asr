#!/usr/bin/env python3
"""Unit tests for transcribe_podcast.py (no external dependencies)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcribe_podcast import (
    ApiError,
    AudioError,
    DependencyError,
    FORMATTERS,
    TokenExpiredError,
    TranscriptionError,
    _detect_asr_bin,
    _detect_model_dir,
    check_env,
    format_output,
    format_srt,
    format_txt,
    get_episode_list,
    get_episode_detail,
    search_episodes,
)


MOCK_EPISODE = {
    "title": "测试单集标题",
    "eid": "test123",
    "podcast": {"title": "测试播客"},
    "pubDate": "2026-05-05T10:00:00Z",
    "duration": 3661,
    "playCount": 1000,
    "media": {"source": {"url": "https://example.com/audio.m4a"}, "size": 10485760},
}


class TestExceptionHierarchy(unittest.TestCase):
    def test_base_exception(self):
        with self.assertRaises(TranscriptionError):
            raise TranscriptionError("test")

    def test_api_error(self):
        with self.assertRaises(ApiError):
            raise ApiError("test", status_code=500)

    def test_token_expired_is_api_error(self):
        with self.assertRaises(ApiError):
            raise TokenExpiredError("expired", status_code=401)

    def test_dependency_error(self):
        with self.assertRaises(TranscriptionError):
            raise DependencyError("missing")

    def test_audio_error(self):
        with self.assertRaises(TranscriptionError):
            raise AudioError("failed")


class TestFormatMarkdown(unittest.TestCase):
    def test_basic_format(self):
        result = format_output(MOCK_EPISODE, "转录文本内容")
        self.assertIn("# 测试单集标题", result)
        self.assertIn("**节目**: 测试播客", result)
        self.assertIn("**日期**: 2026-05-05", result)
        self.assertIn("**时长**: 61分1秒", result)
        self.assertIn("**播放量**: 1,000", result)
        self.assertIn("## 转录文本", result)
        self.assertIn("转录文本内容", result)

    def test_no_play_count(self):
        ep = {**MOCK_EPISODE, "playCount": 0}
        result = format_output(ep, "text")
        self.assertNotIn("播放量", result)

    def test_missing_fields(self):
        result = format_output({}, "text")
        self.assertIn("未知标题", result)
        self.assertIn("未知节目", result)


class TestFormatSrt(unittest.TestCase):
    def test_basic_srt(self):
        result = format_srt(MOCK_EPISODE, "第一句话。第二句话。第三句话。")
        self.assertIn("-->", result)
        self.assertIn("第一句话", result)
        # Should have 3 numbered entries
        self.assertTrue(result.startswith("1\n"))
        self.assertIn("\n\n2\n", result)
        self.assertIn("\n\n3\n", result)

    def test_empty_transcript(self):
        result = format_srt({"duration": 120}, "")
        self.assertIn("-->", result)

    def test_zero_duration(self):
        result = format_srt({"duration": 0}, "一些文字。")
        self.assertIn("一些文字", result)


class TestFormatTxt(unittest.TestCase):
    def test_basic_txt(self):
        result = format_txt(MOCK_EPISODE, "转录内容")
        self.assertEqual(result.split("\n\n")[0], "测试单集标题 | 测试播客 | 2026-05-05")
        self.assertIn("转录内容", result)

    def test_no_podcast(self):
        result = format_txt({"title": "T"}, "text")
        self.assertTrue(result.startswith("T"))


class TestFormattersRegistry(unittest.TestCase):
    def test_all_formats_registered(self):
        self.assertEqual(set(FORMATTERS.keys()), {"markdown", "srt", "txt"})

    def test_each_formatter_callable(self):
        for name, fn in FORMATTERS.items():
            result = fn(MOCK_EPISODE, "test text")
            self.assertIsInstance(result, str, f"Formatter {name} did not return str")


class TestPathDetection(unittest.TestCase):
    def test_model_dir_env_override(self):
        with patch.dict(os.environ, {"QWEN3_ASR_MODEL_DIR": "/fake/path"}):
            with patch.object(Path, "exists", return_value=True):
                self.assertEqual(_detect_model_dir(), "/fake/path")

    def test_asr_bin_env_override(self):
        with patch.dict(os.environ, {"QWEN3_ASR_BIN": "/fake/bin"}):
            with patch.object(Path, "exists", return_value=True):
                self.assertEqual(_detect_asr_bin(), "/fake/bin")


class TestApiFunctions(unittest.TestCase):
    @patch("transcribe_podcast.api")
    def test_search_episodes(self, mock_api):
        mock_api.return_value = {
            "data": {"data": [
                {"type": "EPISODE", "eid": "1", "title": "Ep1"},
                {"type": "PODCAST", "pid": "p1"},
                {"type": "EPISODE", "eid": "2", "title": "Ep2"},
            ]}
        }
        result = search_episodes("token", "test", limit=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["eid"], "1")

    @patch("transcribe_podcast.api")
    def test_get_episode_detail(self, mock_api):
        mock_api.return_value = {"data": {"data": {"eid": "abc", "title": "Test"}}}
        result = get_episode_detail("token", "abc")
        self.assertEqual(result["title"], "Test")

    @patch("transcribe_podcast.api")
    def test_get_episode_list(self, mock_api):
        mock_api.return_value = {"data": {"data": [
            {"eid": "1"}, {"eid": "2"}, {"eid": "3"},
        ]}}
        result = get_episode_list("token", "pid", count=2)
        self.assertEqual(len(result), 2)


class TestCheckEnv(unittest.TestCase):
    @patch("transcribe_podcast.shutil.which")
    @patch("transcribe_podcast.Path.exists")
    def test_check_env_all_missing(self, mock_exists, mock_which):
        mock_which.return_value = None
        mock_exists.return_value = False
        result = check_env(None)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
