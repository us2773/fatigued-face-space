import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fatigued_face_space.nodes.openface.v1.nodes import (
    get_openface_from_csv_timeseries,
    select_oldest_movie,
)
from fatigued_face_space.nodes.cleansing.v1.nodes import (
    get_cleansed_openface_timeseries,
)


class OneMovieIOTests(unittest.TestCase):
    def test_selects_oldest_unprocessed_movie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            movies, results = root / "movies", root / "results"
            movies.mkdir()
            results.mkdir()
            for name, timestamp in (("completed", 1), ("older", 2), ("newer", 3)):
                path = movies / f"{name}.mp4"
                path.touch()
                os.utime(path, (timestamp, timestamp))
            (results / "completed.parquet").touch()
            selected = select_oldest_movie(str(movies), str(results), "/*.mp4", "/*.parquet")
            self.assertEqual(selected, ["older"])

    def test_no_pending_movie_stops_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                select_oldest_movie(directory, directory, "/*.mp4", "/*.parquet")

    def test_csv_import_keeps_unselected_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected.csv"
            unrelated = root / "unrelated.csv"
            selected.write_text("frame\n1\n", encoding="utf-8")
            unrelated.write_text("frame\n2\n", encoding="utf-8")
            frame = object()
            with patch("fatigued_face_space.nodes.openface.v1.nodes.pd.read_csv", return_value=frame):
                output = get_openface_from_csv_timeseries(["selected"], str(root))
            self.assertEqual(output, {"selected": frame})
            self.assertFalse(selected.exists())
            self.assertTrue(unrelated.exists())

    def test_missing_csv_keeps_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            unrelated = Path(directory) / "unrelated.csv"
            unrelated.touch()
            with self.assertRaises(RuntimeError):
                get_openface_from_csv_timeseries(["missing"], directory)
            self.assertTrue(unrelated.exists())

    def test_failed_csv_read_keeps_input(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.csv"
            selected.touch()
            with patch(
                "fatigued_face_space.nodes.openface.v1.nodes.pd.read_csv",
                side_effect=ValueError("invalid CSV"),
            ):
                with self.assertRaises(ValueError):
                    get_openface_from_csv_timeseries(["selected"], directory)
            self.assertTrue(selected.exists())

    def test_cleansing_does_not_load_other_partitions(self):
        frame = MagicMock()
        frame.__getitem__.return_value = frame
        selected = MagicMock(return_value=frame)
        unrelated = MagicMock(side_effect=AssertionError("unselected partition loaded"))
        output = get_cleansed_openface_timeseries(
            {"selected": selected, "unrelated": unrelated}, ["selected"]
        )
        self.assertEqual(list(output), ["selected"])
        selected.assert_called_once_with()
        unrelated.assert_not_called()

    def test_cleansing_without_selection_preserves_batch_behavior(self):
        frame = MagicMock()
        frame.__getitem__.return_value = frame
        loaders = {name: MagicMock(return_value=frame) for name in ("first", "second")}
        self.assertEqual(set(get_cleansed_openface_timeseries(loaders)), set(loaders))
        for loader in loaders.values():
            loader.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
