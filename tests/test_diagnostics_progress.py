from __future__ import annotations

import logging
import unittest

from continuous_tokenizer.runtime.progress import ProgressTracker, log_event


class ProgressLoggingTests(unittest.TestCase):
    def test_progress_is_structured_and_bounded(self) -> None:
        tracker = ProgressTracker(
            "vocabulary_batches",
            100,
            {"epoch": 3},
            updates=4,
        )

        with self.assertLogs("continuous_tokenizer.progress", logging.INFO) as captured:
            for completed in range(1, 101):
                tracker.update(completed)

        messages = [record.getMessage() for record in captured.records]
        self.assertEqual(len(messages), 5)
        for completed, message in zip((1, 25, 50, 75, 100), messages, strict=True):
            self.assertIn('event="progress"', message)
            self.assertIn('phase="vocabulary_batches"', message)
            self.assertIn(f"completed={completed}", message)
            self.assertIn("total=100", message)
            self.assertIn("eta_seconds=", message)
            self.assertIn("epoch=3", message)

    def test_log_event_serializes_fields_unambiguously(self) -> None:
        with self.assertLogs("continuous_tokenizer.progress", logging.INFO) as captured:
            log_event("stage_started", stage="load model", cached=False)

        self.assertEqual(
            captured.records[0].getMessage(),
            'event="stage_started" stage="load model" cached=false',
        )

    def test_progress_rejects_invalid_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ProgressTracker("invalid", -1)
        with self.assertRaisesRegex(ValueError, "positive"):
            ProgressTracker("invalid", 1, updates=0)
        with self.assertRaisesRegex(ValueError, "within"):
            ProgressTracker("invalid", 1).update(0)


if __name__ == "__main__":
    unittest.main()
