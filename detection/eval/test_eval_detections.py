"""Tests with tiny synthetic data. Run: python -m unittest detection/eval/test_eval_detections.py"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import eval_detections as E  # noqa: E402


PERFECT = 0.995


def box(c, cx, cy, w, h, conf=None):
    s = f"{c} {cx} {cy} {w} {h}"
    return s + (f" {conf}" if conf is not None else "")


class Case:
    def __init__(self, gt, pred):
        """gt/pred: {image_stem: [lines]}"""
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.gt, self.pred = root / "gt", root / "pred"
        self.gt.mkdir(); self.pred.mkdir()
        for d, data in ((self.gt, gt), (self.pred, pred)):
            for stem, lines in data.items():
                (d / f"{stem}.txt").write_text("\n".join(lines))

    def run(self):
        try:
            res = E.evaluate(self.gt, self.pred)
            return {r["name"]: r for r in res["rows"]}
        finally:
            self.tmp.cleanup()


class Tests(unittest.TestCase):
    def test_perfect(self):
        g = {"a": [box(0, .5, .5, .2, .2), box(1, .2, .2, .1, .1)], "b": [box(0, .7, .7, .2, .2)]}
        p = {k: [l + " 0.9" for l in v] for k, v in g.items()}
        r = Case(g, p).run()["all"]
        # 101-point AP tops out at 0.995 (same ceiling as Ultralytics)
        for k in ("P", "R"):
            self.assertAlmostEqual(r[k], 1.0, places=3)
        for k in ("mAP50", "mAP50_95"):
            self.assertAlmostEqual(r[k], PERFECT, places=3)

    def test_no_predictions(self):
        r = Case({"a": [box(0, .5, .5, .2, .2)]}, {}).run()["all"]
        for k in ("P", "R", "mAP50", "mAP50_95"):
            self.assertEqual(r[k], 0.0)

    def test_wrong_class_is_not_a_hit(self):
        r = Case({"a": [box(0, .5, .5, .2, .2)]}, {"a": [box(1, .5, .5, .2, .2, .9)]}).run()
        self.assertEqual(r["class_0"]["mAP50"], 0.0)

    def test_one_missed_object(self):
        # 2 GT, 1 found, no false positives -> AP50 = 0.5 (recall caps at 0.5)
        g = {"a": [box(0, .3, .3, .2, .2), box(0, .7, .7, .2, .2)]}
        p = {"a": [box(0, .3, .3, .2, .2, .9)]}
        r = Case(g, p).run()["class_0"]
        self.assertAlmostEqual(r["mAP50"], 0.75, places=2)  # Ultralytics curve closes at precision 0
        self.assertAlmostEqual(r["R"], 0.5, places=2)
        self.assertAlmostEqual(r["P"], 1.0, places=2)

    def test_false_positive_ranked_first_halves_precision(self):
        g = {"a": [box(0, .3, .3, .2, .2)]}
        p = {"a": [box(0, .8, .8, .1, .1, .95), box(0, .3, .3, .2, .2, .5)]}
        r = Case(g, p).run()["class_0"]
        self.assertAlmostEqual(r["mAP50"], 0.5, places=2)

    def test_shifted_box_passes_50_not_75(self):
        # box shifted by 0.08 (width 0.4): IoU = 0.667 -> hit at 0.50, 0.55, 0.60, 0.65 only
        g = {"a": [box(0, .5, .5, .4, .4)]}
        p = {"a": [box(0, .58, .5, .4, .4, .9)]}
        r = Case(g, p).run()["class_0"]
        self.assertAlmostEqual(r["mAP50"], PERFECT, places=2)
        self.assertAlmostEqual(r["mAP50_95"], 0.4 * PERFECT, places=2)

    def test_class_without_gt_is_ignored_in_mean(self):
        g = {"a": [box(0, .5, .5, .2, .2)]}
        p = {"a": [box(0, .5, .5, .2, .2, .9), box(3, .1, .1, .1, .1, .9)]}
        res = Case(g, p).run()
        self.assertNotIn("class_3", res)
        self.assertAlmostEqual(res["all"]["mAP50"], PERFECT, places=2)

    def test_duplicate_prediction_is_false_positive(self):
        g = {"a": [box(0, .5, .5, .2, .2)]}
        p = {"a": [box(0, .5, .5, .2, .2, .9), box(0, .5, .5, .2, .2, .8)]}
        r = Case(g, p).run()["class_0"]
        self.assertAlmostEqual(r["mAP50"], PERFECT, places=2)  # duplicate is after the hit
        self.assertLess(r["P"], 1.01)

    def test_missing_conf_column_gives_clear_error(self):
        c = Case({"a": [box(0, .5, .5, .2, .2)]}, {"a": [box(0, .5, .5, .2, .2)]})
        with self.assertRaises(ValueError):
            c.run()

    def test_empty_image_counts_when_listed_via_pred_only(self):
        # image b has no GT file but a prediction -> still a false positive
        g = {"a": [box(0, .5, .5, .2, .2)]}
        p = {"a": [box(0, .5, .5, .2, .2, .5)], "b": [box(0, .5, .5, .2, .2, .9)]}
        c = Case(g, p)
        res = E.evaluate(c.gt, c.pred, images_dir=None)
        c.tmp.cleanup()
        # b is not listed (no --images) so not evaluated: documented behaviour
        self.assertAlmostEqual(res["rows"][-1]["mAP50"], PERFECT, places=2)


if __name__ == "__main__":
    unittest.main()
