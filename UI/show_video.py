from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.videoplayer import VideoPlayer


class MyW(Widget):
    def __init__(self, video_source, **kwargs):
        super(MyW, self).__init__(**kwargs)
        player = VideoPlayer(
            source=video_source, state='play',
            options={'allow_stretch': True},
            size=(600, 600))
        self.add_widget(player)


class e5App(App):
    def __init__(self, video_source):
        super(e5App, self).__init__()
        self.video_source = video_source

    def build(self):
        return MyW(self.video_source)


# if __name__ == '__main__':
#     video_source = '../asset/pro.mp4'
#     e5App(video_source).run()