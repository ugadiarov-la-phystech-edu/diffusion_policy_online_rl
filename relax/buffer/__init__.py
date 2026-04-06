from relax.buffer.base import Buffer
from relax.buffer.tree import TreeBuffer as TreeBuffer
from relax.buffer.frame_stack import FrameStackBuffer as FrameStackBuffer
from relax.utils.experience import Experience

ExperienceBuffer = Buffer[Experience]
