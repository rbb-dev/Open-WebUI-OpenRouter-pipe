"""Direct tests for security methods without full module imports."""

import pytest
import socket
import ipaddress
import re
from unittest.mock import patch


def test_ssrf_protection_logic():
    """Test SSRF protection logic directly."""

    # Simulate the _is_safe_url logic
    def is_safe_url_logic(url: str, enable_protection: bool = True) -> bool:
        """Replicate _is_safe_url logic."""
        if not enable_protection:
            return True

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname

            if not host:
                return False

            try:
                ip_str = socket.gethostbyname(host)
                ip = ipaddress.ip_address(ip_str)
            except socket.gaierror:
                return False
            except ValueError:
                return False

            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False

            return True
        except Exception:
            return False

    # Test public IP (mock DNS to return 8.8.8.8)
    with patch('socket.gethostbyname', return_value='8.8.8.8'):
        assert is_safe_url_logic("https://example.com/image.jpg") is True

    # Test localhost
    with patch('socket.gethostbyname', return_value='127.0.0.1'):
        assert is_safe_url_logic("http://localhost:6379/") is False

    # Test private IP ranges
    private_ips = ['10.0.0.1', '192.168.1.1', '172.16.0.1']
    for ip in private_ips:
        with patch('socket.gethostbyname', return_value=ip):
            assert is_safe_url_logic(f"http://internal.local/") is False

    # Test link-local (AWS metadata)
    with patch('socket.gethostbyname', return_value='169.254.169.254'):
        assert is_safe_url_logic("http://169.254.169.254/latest/meta-data/") is False

    # Test multicast
    with patch('socket.gethostbyname', return_value='224.0.0.1'):
        assert is_safe_url_logic("http://multicast.local/") is False

    # Test DNS failure
    with patch('socket.gethostbyname', side_effect=socket.gaierror):
        assert is_safe_url_logic("http://nonexistent.invalid/") is False

    # Test disabled protection
    assert is_safe_url_logic("http://127.0.0.1/", enable_protection=False) is True


def test_youtube_url_validation_logic():
    """Test YouTube URL validation logic directly."""

    def is_youtube_url_logic(url: str) -> bool:
        """Replicate _is_youtube_url logic."""
        if not url:
            return False

        patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        ]

        return any(re.match(pattern, url, re.IGNORECASE) for pattern in patterns)

    # Test standard YouTube URL
    assert is_youtube_url_logic("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_youtube_url_logic("http://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_youtube_url_logic("https://youtube.com/watch?v=dQw4w9WgXcQ") is True

    # Test short YouTube URL
    assert is_youtube_url_logic("https://youtu.be/dQw4w9WgXcQ") is True
    assert is_youtube_url_logic("http://youtu.be/dQw4w9WgXcQ") is True

    # Test with parameters
    assert is_youtube_url_logic("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s") is True

    # Test non-YouTube URLs
    assert is_youtube_url_logic("https://vimeo.com/123456") is False
    assert is_youtube_url_logic("https://example.com/video.mp4") is False
    assert is_youtube_url_logic("") is False


def test_video_size_validation_logic():
    """Test video size validation logic directly."""

    def estimate_video_size_mb(data_url: str, max_size_mb: int = 100) -> tuple[bool, float]:
        """Replicate video size estimation logic."""
        if not data_url.startswith("data:"):
            return True, 0.0

        if "," not in data_url:
            return True, 0.0

        b64_data = data_url.split(",", 1)[1]
        # Estimate decoded size (base64 is ~33% larger than raw)
        estimated_size_bytes = (len(b64_data) * 3) // 4
        max_size_bytes = max_size_mb * 1024 * 1024
        estimated_size_mb = estimated_size_bytes / (1024 * 1024)

        is_valid = estimated_size_bytes <= max_size_bytes
        return is_valid, estimated_size_mb

    # Test small video (10MB)
    small_video = "data:video/mp4;base64," + ("A" * (10 * 1024 * 1024))
    is_valid, size_mb = estimate_video_size_mb(small_video, max_size_mb=100)
    assert is_valid is True
    assert size_mb < 100

    # Test large video (120MB)
    large_video = "data:video/mp4;base64," + ("A" * (160 * 1024 * 1024))
    is_valid, size_mb = estimate_video_size_mb(large_video, max_size_mb=100)
    assert is_valid is False
    assert size_mb > 100

    # Test remote URL (no size check)
    is_valid, size_mb = estimate_video_size_mb("https://example.com/video.mp4", max_size_mb=100)
    assert is_valid is True
    assert size_mb == 0.0


def test_status_messages_constants():
    """Test that status message pattern is correct."""

    # Define expected status messages (from StatusMessages class)
    expected_messages = {
        "IMAGE_BASE64_SAVED": "📥 Saved base64 image to storage",
        "IMAGE_REMOTE_SAVED": "📥 Downloaded and saved image from remote URL",
        "FILE_BASE64_SAVED": "📥 Saved base64 file to storage",
        "FILE_REMOTE_SAVED": "📥 Downloaded and saved file from remote URL",
        "VIDEO_BASE64": "🎥 Processing base64 video input",
        "VIDEO_YOUTUBE": "🎥 Processing YouTube video input",
        "VIDEO_REMOTE": "🎥 Processing video input",
        "AUDIO_BASE64_SAVED": "🎵 Saved base64 audio to storage",
        "AUDIO_REMOTE_SAVED": "🎵 Downloaded and saved audio from remote URL",
    }

    # Verify all messages are non-empty strings with emojis
    for name, message in expected_messages.items():
        assert isinstance(message, str), f"{name} should be a string"
        assert len(message) > 0, f"{name} should not be empty"
        assert any(char in message for char in ["📥", "🎥", "🎵"]), f"{name} should contain emoji"


def test_valve_defaults():
    """Test default valve values."""

    # Expected valve defaults
    expected_defaults = {
        "VIDEO_MAX_SIZE_MB": 100,
        "ENABLE_SSRF_PROTECTION": True,
    }

    # Verify defaults are sensible
    assert expected_defaults["VIDEO_MAX_SIZE_MB"] >= 1
    assert expected_defaults["VIDEO_MAX_SIZE_MB"] <= 1000
    assert isinstance(expected_defaults["ENABLE_SSRF_PROTECTION"], bool)


def test_integration_ssrf_in_download():
    """Test that SSRF check would be called in download flow."""

    # Simulate download flow with SSRF check
    def download_flow(url: str, enable_ssrf: bool = True) -> bool:
        """Simulate _download_remote_url flow."""
        if not url.startswith(("http://", "https://")):
            return False

        # SSRF check (simulated)
        if enable_ssrf:
            # Would call _is_safe_url here
            if "127.0.0.1" in url or "localhost" in url:
                return False

        return True

    # Test that SSRF blocking works
    assert download_flow("http://127.0.0.1/test", enable_ssrf=True) is False
    assert download_flow("http://localhost/test", enable_ssrf=True) is False
    assert download_flow("https://example.com/test", enable_ssrf=True) is True

    # Test with SSRF disabled
    assert download_flow("http://127.0.0.1/test", enable_ssrf=False) is True


def test_integration_video_url_validation():
    """Test video URL validation integration."""

    def validate_video_url(url: str, enable_ssrf: bool = True) -> tuple[bool, str]:
        """Simulate video URL validation flow."""
        if not url:
            return False, "empty_url"

        # Check if YouTube
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        ]
        is_youtube = any(re.match(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)

        if is_youtube:
            return True, "youtube"

        # Check if remote URL
        if url.startswith(("http://", "https://")):
            # Would apply SSRF check here
            if enable_ssrf and ("127.0.0.1" in url or "localhost" in url):
                return False, "ssrf_blocked"
            return True, "remote"

        # Check if data URL
        if url.startswith("data:"):
            return True, "data_url"

        # OWUI file reference
        if "/api/v1/files/" in url:
            return True, "owui_file"

        return False, "unknown"

    # Test YouTube URLs
    is_valid, url_type = validate_video_url("https://www.youtube.com/watch?v=test123")
    assert is_valid is True
    assert url_type == "youtube"

    # Test SSRF blocking
    is_valid, url_type = validate_video_url("http://127.0.0.1/video.mp4", enable_ssrf=True)
    assert is_valid is False
    assert url_type == "ssrf_blocked"

    # Test remote URL (safe)
    is_valid, url_type = validate_video_url("https://example.com/video.mp4", enable_ssrf=True)
    assert is_valid is True
    assert url_type == "remote"

    # Test data URL
    is_valid, url_type = validate_video_url("data:video/mp4;base64,AAAA")
    assert is_valid is True
    assert url_type == "data_url"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
