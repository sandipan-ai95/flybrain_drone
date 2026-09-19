"""Vision: camera stub, visual encoder, optical flow."""
from .camera import Camera, SyntheticCamera, CameraFrame
from .encoder import VisualEncoder, EncoderConfig
from .optical_flow import frame_difference

__all__ = ["Camera", "SyntheticCamera", "CameraFrame",
           "VisualEncoder", "EncoderConfig", "frame_difference"]
