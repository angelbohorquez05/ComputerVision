from src.evaluate_detector import (
    _average_precision,
    _load_ground_truth,
    _match_predictions,
)
from src.fusion import BBox


def test_match_predictions_perfect_hit():
    gt = [BBox(0.30, 0.30, 0.60, 0.60)]
    preds = [(0.9, BBox(0.31, 0.31, 0.61, 0.61))]   # IoU well above 0.5
    assert _match_predictions(preds, gt) == [1]


def test_match_predictions_duplicate_is_false_positive():
    # Two predictions on one GT box: the best is a TP, the extra is an FP,
    # because each GT box can be matched only once.
    gt = [BBox(0.30, 0.30, 0.60, 0.60)]
    preds = [
        (0.9, BBox(0.30, 0.30, 0.60, 0.60)),   # exact → TP
        (0.8, BBox(0.31, 0.31, 0.61, 0.61)),   # overlaps same GT → FP
    ]
    assert _match_predictions(preds, gt) == [1, 0]


def test_match_predictions_far_box_is_false_positive():
    gt = [BBox(0.30, 0.30, 0.60, 0.60)]
    preds = [(0.9, BBox(0.05, 0.05, 0.15, 0.15))]   # no overlap
    assert _match_predictions(preds, gt) == [0]


def test_average_precision_perfect_ranking():
    # Every prediction is a TP and every GT is found → AP = 1.0
    assert _average_precision([1, 1, 1], n_ground_truth=3) == 1.0


def test_average_precision_no_ground_truth_is_zero():
    assert _average_precision([0, 0], n_ground_truth=0) == 0.0


def test_load_ground_truth_parses_yolo_format(tmp_path):
    label = tmp_path / "img.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n")
    boxes = _load_ground_truth(label)
    assert len(boxes) == 1
    b = boxes[0]
    assert abs(b.x1 - 0.30) < 1e-9 and abs(b.x2 - 0.70) < 1e-9
    assert abs(b.y1 - 0.40) < 1e-9 and abs(b.y2 - 0.60) < 1e-9


def test_load_ground_truth_missing_file_is_empty(tmp_path):
    assert _load_ground_truth(tmp_path / "nope.txt") == []
