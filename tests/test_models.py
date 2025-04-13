import unittest
from unittest.mock import patch, MagicMock
from .utils import assert_almost_equal  # Fix import path
from ..app.models.caught_behind_model import CaughtBehindModel
from ..app.models.runout_stumping_model import RunoutStumpingModel

class TestCaughtBehindModel(unittest.TestCase):
    @patch('..app.models.caught_behind_model.cv2.VideoCapture')
    @patch('..app.models.caught_behind_model.YOLO')
    def test_process_video_spike_detected(self, mock_yolo, mock_video_capture):
        # Mock YOLO model
        mock_spike_model = MagicMock()
        mock_edge_model = MagicMock()
        mock_yolo.side_effect = [mock_spike_model, mock_edge_model]

        # Mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, MagicMock())

        # Mock spike detection
        mock_spike_model.return_value = [MagicMock(boxes=[MagicMock(cls=[0], xyxy=[[10, 10, 50, 50]])])]
        mock_spike_model.names = {0: "spike"}
        mock_edge_model.return_value = [MagicMock(boxes=[MagicMock(cls=[1], xyxy=[[20, 20, 60, 60]])])]
        mock_edge_model.names = {1: "edge"}

        model = CaughtBehindModel()
        result, frames = model.process_video("dummy_video_path")

        self.assertTrue(result)
        self.assertEqual(len(frames), 1)
        self.assertIn('filename', frames[0])
        self.assertIn('label', frames[0])

    @patch('..app.models.caught_behind_model.cv2.VideoCapture')
    @patch('..app.models.caught_behind_model.YOLO')
    def test_process_video_no_spike_detected(self, mock_yolo, mock_video_capture):
        # Mock YOLO model
        mock_spike_model = MagicMock()
        mock_edge_model = MagicMock()
        mock_yolo.side_effect = [mock_spike_model, mock_edge_model]

        # Mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, MagicMock())

        # Mock no spike detection
        mock_spike_model.return_value = []
        mock_spike_model.names = {0: "spike"}

        model = CaughtBehindModel()
        result, frames = model.process_video("dummy_video_path")

        self.assertFalse(result)
        self.assertEqual(len(frames), 0)

class TestRunoutStumpingModel(unittest.TestCase):
    @patch('..app.models.runout_stumping_model.cv2.VideoCapture')
    @patch('..app.models.runout_stumping_model.YOLO')
    def test_process_video_stumps_hit(self, mock_yolo, mock_video_capture):
        # Mock YOLO model
        mock_stumps_model = MagicMock()
        mock_batsman_model = MagicMock()
        mock_yolo.side_effect = [mock_stumps_model, mock_batsman_model]

        # Mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, MagicMock())

        # Mock stumps detection
        mock_stumps_model.return_value = [MagicMock(boxes=[MagicMock(cls=1)])]
        mock_batsman_model.return_value = []

        model = RunoutStumpingModel()
        result, frames = model.process_video("dummy_video_path")

        self.assertTrue(result)
        self.assertEqual(len(frames), 1)
        self.assertIn('filename', frames[0])
        self.assertIn('label', frames[0])

    @patch('..app.models.runout_stumping_model.cv2.VideoCapture')
    @patch('..app.models.runout_stumping_model.YOLO')
    def test_process_video_no_stumps_hit(self, mock_yolo, mock_video_capture):
        # Mock YOLO model
        mock_stumps_model = MagicMock()
        mock_batsman_model = MagicMock()
        mock_yolo.side_effect = [mock_stumps_model, mock_batsman_model]

        # Mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, MagicMock())

        # Mock no stumps detection
        mock_stumps_model.return_value = []

        model = RunoutStumpingModel()
        result, frames = model.process_video("dummy_video_path")

        self.assertFalse(result)
        self.assertEqual(len(frames), 0)

if __name__ == '__main__':
    unittest.main()
