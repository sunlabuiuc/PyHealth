from unittest.mock import patch

from tests.base import BaseTestCase
from pyhealth import _ensure_cache_path


class TestCachePath(BaseTestCase):
    def test_ensure_cache_path_falls_back_when_preferred_is_unwritable(self):
        preferred_path = "/preferred/cache/path"
        fallback_path = "/tmp/pyhealth-cache-test"
        attempts = []

        def fake_makedirs(path, exist_ok=False):
            attempts.append((path, exist_ok))
            if path == preferred_path:
                raise PermissionError("permission denied")

        with patch("pyhealth.os.makedirs", side_effect=fake_makedirs):
            resolved_path = _ensure_cache_path(preferred_path, fallback_path)

        self.assertEqual(fallback_path, resolved_path)
        self.assertEqual(
            [(preferred_path, True), (fallback_path, True)],
            attempts,
        )

    def test_ensure_cache_path_raises_with_context_when_all_candidates_fail(self):
        preferred_path = "/preferred/cache/path"
        fallback_path = "/fallback/cache/path"

        def always_fail(path, exist_ok=False):
            raise PermissionError(f"no access: {path}")

        with patch("pyhealth.os.makedirs", side_effect=always_fail):
            with self.assertRaises(OSError) as err:
                _ensure_cache_path(preferred_path, fallback_path)

        error_message = str(err.exception)
        self.assertIn(preferred_path, error_message)
        self.assertIn(fallback_path, error_message)
