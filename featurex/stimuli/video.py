from __future__ import division
from featurex.stimuli import DynamicStim
from featurex.stimuli.image import ImageStim
from featurex.core import Timeline, Event
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd


class VideoFrameStim(ImageStim):

    ''' A single frame of video. '''

    def __init__(self, video, frame_num, duration=None, filename=None, data=None):
        super(VideoFrameStim, self).__init__(filename, data)
        self.video = video
        self.data = self.video.frames[frame_num]
        self.frame_num = frame_num
        spf = 1. / video.fps
        if duration is None:
            self.duration = spf
        else:
            self.duration = duration
        self.onset = frame_num * spf


class VideoStim(DynamicStim):

    ''' A video. '''

    def __init__(self, filename):

        self.clip = VideoFileClip(filename)
        self.fps = self.clip.fps
        self.width = self.clip.w
        self.height = self.clip.h

        self.frames = [f for f in self.clip.iter_frames()]
        self.n_frames = len(self.frames)

        super(VideoStim, self).__init__(filename)

    def _extract_duration(self):
        self.duration = self.n_frames * 1. / self.fps

    def __iter__(self):
        """ Frame iteration. """
        for i, f in enumerate(self.frames):
            yield VideoFrameStim(self, i, data=f)

    def extract(self, extractors, merge_events=True, **kwargs):
        period = 1. / self.fps
        timeline = Timeline(period=period)
        for ext in extractors:
            # For VideoExtractors, pass the entire stim
            if ext.target.__name__ == self.__class__.__name__:
                events = ext.transform(self, **kwargs)
                for ev in events:
                    timeline.add_event(ev, merge=merge_events)
            # Otherwise, for images, loop over frames
            else:
                c = 0
                for frame in self:
                    if frame.data is not None:
                        event = Event(onset=c * period)
                        event.add_value(ext.transform(frame))
                        timeline.add_event(event, merge=merge_events)
                        c += 1
        return timeline


class DerivedVideoStim(VideoStim):
    """
    VideoStim containing keyframes (for API calls). Each keyframe is associated
    with a duration reflecting the length of its "scene."
    """
    def __init__(self, filename, elements, frame_index=None, history=None):
        super(DerivedVideoStim, self).__init__(filename)
        self.elements = elements
        self.frame_index = frame_index
        self.history = history
        
    def __iter__(self):
        for elem in self.elements:
            yield elem

