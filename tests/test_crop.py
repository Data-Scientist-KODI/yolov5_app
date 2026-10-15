
from crop import select_best_box


class FakeBox:
    """A lightweight substitute for a YOLO detection box."""

    def __init__(self, confidence):
        self.conf = [confidence]


def test_select_best_box():
    # Arrange: create three detections with different confidence scores
    box1 = FakeBox(0.65)
    box2 = FakeBox(0.92)
    box3 = FakeBox(0.78)

    boxes = [box1, box2, box3]

    # Act: select the highest-confidence detection
    best_box, confidence = select_best_box(boxes)

    # Assert: verify that the correct detection was selected
    assert best_box is box2
    assert confidence == 0.92


def test_select_best_box_when_empty():
    # An empty list means no face was detected
    best_box, confidence = select_best_box([])

    assert best_box is None
    assert confidence == -1.0
