#!/usr/bin/env python3

from random import shuffle
import src.util as APL_UTIL

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2

import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import kivy

from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock

from kivy.uix.screenmanager import ScreenManager
from kivy.uix.screenmanager import Screen
from kivy.uix.popup import Popup

from kivy.uix.widget import Widget
from kivy.uix.button import Button

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stacklayout import StackLayout

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg as Figure

from kivy.config import Config

import os

kivy.require('1.10.0')
Window.clearcolor = .85, .85, .85, 1

IMG_SCALE = 64

class Chip(BoxLayout): pass
class ChipInput(Chip):
    def __init__(self, *args, **kwargs):
        super(ChipInput, self).__init__(*args, **kwargs)
        self.callback = lambda x: None

    def submit(self):
        self.callback(self.ids.input.text)
        
class ChipInputAdder(Chip):
    def __init__(self, *args, **kwargs):
        super(ChipInputAdder, self).__init__(*args, **kwargs)
        self.callback = lambda x: None

    def submit(self):
        self.callback(self.ids.input.text)
        self.ids.input.text = ''

class ChipRemovable(Chip):
    def __init__(self, *args, **kwargs):
        super(ChipRemovable, self).__init__(*args, **kwargs)

    def on_press(self):
        self.selected = not self.selected

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.ids.btn_remove.on_touch_down(touch)
        
            self.on_press()
        
    def remove(self):
        pass


class Plot(Widget):
    def __init__(self, *args, **kwargs):
        super(Plot, self).__init__(*args, **kwargs)

        self.orientation = 'vertical'
        self.background_color = 0, 0, 0, 0
        self.size_hint = None, None

        self.fig = plt.figure()
        self.fig.subplots_adjust(bottom=0, left=0, top=1, right=1)

        self.fig.patch.set_facecolor((0, 0, 0, 0))

        self.plot = self.fig.add_subplot(111)
        
    def update(self):
        self.fig.canvas.draw()

class ImagePlot(BoxLayout):
    def __init__(self, *args, **kwargs):
        super(ImagePlot, self).__init__(*args, **kwargs)

        self.orientation = 'vertical'
        self.background_color = 0, 0, 0, 0
        self.size_hint = None, None

        self.fig = plt.figure()
        self.fig.subplots_adjust(bottom=0, left=0, top=1, right=1)

        self.fig.patch.set_facecolor((0, 0, 0, 0))

        
        self.img_plot = self.fig.add_subplot(111)
        self.img_plot.set_axis_off()
        self.set_image(np.zeros((1, 1, 3)))

        self.view_box = patches.Rectangle((0, 0), 0, 0, fill=False)
        self.view_box.set_linestyle('--')
        self.img_plot.add_patch(self.view_box)

        self.figure = Figure(self.fig)
        self.figure.pos_hint = { 'left': 0, 'bottom': 0 }
        self.figure.size_hint = (1, 1)
        self.add_widget(self.figure)

    def update_viewbox(self, x, y, w, h):
        self.view_box.set_xy((x, y))
        self.view_box.set_width(w)
        self.view_box.set_height(h)
        self.fig.canvas.draw()

    def set_image(self, img):
        self.source_img = img
        self.update_image()

    def update_image(self):
        self.img_plot.imshow(self.source_img, interpolation='nearest')
        self.fig.canvas.draw()

    def load_image(self, path):
        self.set_image(plt.imread(path))

class PopupFileLoader(Popup):
    def __init__(self, callback, *args, **kwargs):
        super(PopupFileLoader, self).__init__(*args, **kwargs)
        self.callback = callback

    def selectFile(self, file):
        self.file_path = file[0] if file else None

    def submitFile(self):
        if self.file_path is not None:
            self.callback(self.file_path)
            self.dismiss()

    def cancelFile(self):
        self.file_path = None
        self.dimiss()
        
class ImageLoader(ImagePlot):
    def popup_selectImage(self, callback = lambda: None):
        def loader(file_path):
            try:
                self.load_image(file_path)
                callback()
            except:
                pass

        PopupFileLoader(loader).open()
        
class SpaceStart(Screen): pass

import queue, threading

class SpaceCreateSlide(Screen):    
    class Webcam(object):
        def __init__(self, URL):
            self.cap = cv2.VideoCapture(URL)
            self.q = queue.Queue()

            self.running = threading.Event()
            self.running.set()

            self.thread = threading.Thread(target=self._reader)
            self.thread.daemon = True
            self.thread.start()

        def _reader(self):
            while self.running.is_set():
                ret, frame = self.cap.read()

                if not ret:
                    break
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass

                self.q.put(frame)
            
        def read(self):
            return self.q.get()

        def terminate(self):
            self.running.clear()
            self.thread.join()

    def __init__(self, *args, **kwargs):
        super(SpaceCreateSlide, self).__init__(*args, **kwargs)
       
        self.ip = ''
        self.web_cam_on = False
        self.cam = None

    def set_ip(self, text):
        self.ip = text
        print(self.ip)

    def toggle_webcam(self):
        self.web_cam_on = not self.web_cam_on

        if self.web_cam_on:
            URL = f"http://{self.ip}:8080/video"

            if self.ip:
                print("OPENING URL", URL)
                
                def draw_capture(t):
                    frame = self.cam.read()
                    if frame is not None:
                        self.frame = frame
                        self.ids.plot.set_image(self.frame)
                    print(t)
                    return self.web_cam_on

                self.cam = SpaceCreateSlide.Webcam(URL)
                self.event = Clock.schedule_interval(draw_capture, 2)

            self.ids.btn_start.text = "Stop Webcam"
        else:
            if self.cam is not None:
                self.event.cancel()
                self.cam.terminate()
                self.cam = None

            self.ids.btn_start.text = "Start Webcam"

    def capture(self):
        cv2.imwrite('samples/web.png', self.frame)

class APL_Database:
    path = os.path.join(APL_UTIL.current_dir, 'database')
    samples_path = os.path.join(path, 'samples')
    filters_path = os.path.join(path, 'filters')

    for i in [path, samples_path, filters_path]:
        if not os.path.exists(i):
            os.makedirs(i)

    ID = 0

    @staticmethod
    def saveImage(img, tags):
        for i in tags:
            tag_path = os.path.join(APL_Database.samples_path, f'{i}')
            file_path = os.path.join(tag_path, f'subsample-{APL_Database.ID}.png')
            
            try:
                if not os.path.exists(tag_path):
                    os.makedirs(tag_path)
                
                cv2.imwrite(file_path, img)
            except Exception as ex:
                print(ex)

        APL_Database.ID += 1

    @staticmethod
    def loadTagData(tag, n_begin = 0, N_total = 1000):
        print(f"Loading [{tag}] [", end='')
        tag_path = os.path.join(APL_Database.samples_path, tag)
        
        imgs = []
        
        n = 0
        for path, _, file_names in os.walk(tag_path):
            for file in file_names:    
                if n_begin < n:
                    if n % 100 == 0:
                        print(n, end=':')
                
                    try:
                        imgs.append(plt.imread(os.path.join(path, file)))
                    except:
                        pass
                if n_begin + N_total <= n:
                    break

                n += 1
        print(']')
        return imgs

    @staticmethod
    def getAllTags():
        tags = []
        for _, dir_names, _ in os.walk(APL_Database.samples_path):
            for name in dir_names:
                name = name[:-3]
                if name and name not in tags:
                    tags.append(name)
        return tags

class Filter(object):
    POSITIVE = 0
    NEGATIVE = 1

    def preprocess(self, img):
        _img = cv2.resize(img, (self.scale, self.scale))

        if np.amax(_img) > 1:
            return _img / 255
        return _img[:,:,:3]


    def __init__(self, tag=''):
        self.scale = 64

        self.tag = tag
        self.path = os.path.join(APL_Database.filters_path, self.tag)

        self.key = { Filter.POSITIVE: 'positive', Filter.NEGATIVE: 'negative'}

        self.model = self.MakeV1Model()

        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.data_queue = queue.Queue()

    def MakeV1Model(self):
        return keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu',
                input_shape=(self.scale, self.scale, 3)),
            keras.layers.MaxPool2D((2, 2)),
            
            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            
            keras.layers.Dense(2, activation='softmax')
        ])
    def MakeV2Model(self):
        return keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu',
                input_shape=(self.scale, self.scale, 3)),
            keras.layers.MaxPool2D((2, 2)),
            
            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            
            keras.layers.Dense(2, activation='softmax')
        ])
        
    def get_batch(self, n, N):
        pos_img = APL_Database.loadTagData(self.tag + '+ve', n, N)
        neg_img = APL_Database.loadTagData(self.tag + '-ve', n, N)

        data = []

        N = min(len(pos_img), len(neg_img))
        for i in range(N):
            data.append((self.preprocess(pos_img[i]), Filter.POSITIVE))
            data.append((self.preprocess(neg_img[i]), Filter.NEGATIVE))

        if data:
            return (list(t) for t in zip(*data))
        else:
            return ([], [])

    def train_model_daemon(self, plot = None):
        print('=' * 10, 'Loading', 10 * '=')

        N = 20000
        epochs = 1
        batch_size = 200

        for i in range(0, N - batch_size, batch_size):
            imgs, labels = self.get_batch(i, batch_size)

            if imgs:
                fixed_point = round(len(imgs) * 0.9)

                train_imgs = np.array(imgs[:fixed_point])
                train_labels = np.array(labels[:fixed_point])

                test_imgs = np.array(imgs[fixed_point:])
                test_labels = np.array(labels[fixed_point:])

                img = train_imgs[0]
                print('=' * 10, 'DataFormat', 10 * '=')
                print(f" - Train Imgs[{train_imgs.shape}] Label[{train_labels.shape}]")
                print(f" - Test Imgs[{test_imgs.shape}] Label[{test_labels.shape}]")
                print(f" - Image[{img.shape}] min {np.amin(img)} max {np.amax(img)}")

                train_gen = ImageDataGenerator(
                    samplewise_std_normalization = True,
                    brightness_range=(.0, .5),
                    channel_shift_range=.3,
                    horizontal_flip=True,
                    vertical_flip=True
                )
                train_gen.fit(train_imgs)

                test_gen = ImageDataGenerator(
                    samplewise_std_normalization = True,
                    brightness_range=(.0, .5),
                    channel_shift_range=.3,
                    horizontal_flip=True,
                    vertical_flip=True    
                )
                test_gen.fit(test_imgs)
                
                self.model.fit_generator(
                    train_gen.flow(train_imgs, train_labels), 
                    steps_per_epoch=len(train_imgs),
                    epochs=epochs,
                    validation_data=test_gen.flow(test_imgs, test_labels),
                    validation_steps=20,
                )

                #self.model.fit(train_imgs, train_labels, epochs=30)
                #loss, acc = self.model.evaluate(test_imgs, test_labels)
            else:
                break

    def train_model_multi(self):
        self.queue = queue.Queue()
        self.train_model_daemon()

    def train_model(self):
        self.queue = queue.Queue()
        self.train_model_daemon()

    def evaluate(self):
        print("=" * 10, "Evaluate", "=" * 10)

        imgs, labels = self.get_batch(0, 50)
        self.model.evaluate(np.array(imgs), np.array(labels), verbose=2)

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.model.save_weights(os.path.join(self.path, 'state.ckpt'))
        print(f"Saved {self.tag} filter")
        return self

    def load(self, path):
        latest = tensorflow.train.latest_checkpoint(path)
        print(latest)

        self.model.load_weights(os.path.join(path, 'state.ckpt'))
        self.evaluate()
        return self


    def train(self, plot = None):
        #Clock.schedule_once(lambda t: self.train_model(plot), 1)
        Clock.schedule_once(lambda t: self.train_model(), 1)

    def loadData(self):
        self.imgs = APL_Database.loadTagData(self.tag + '+ve')
        self.neg_imgs = APL_Database.loadTagData(self.tag + '-ve')
        return self

    def predict(self, img):
        return self.model.predict(np.array([self.preprocess(img)]))[0]

    @staticmethod
    def loadAllFilters():
        filters = []

        for path, sub_dir, _ in os.walk(APL_Database.filters_path):
            for tag_name in sub_dir:
                folder_path = os.path.join(path, tag_name)
                filters.append(Filter(tag_name).load(folder_path))

        return filters

class SpaceAnalyze(Screen): 
    class FilterApply(StackLayout):
        class Analytics(object):
            def __init__(self):
                self.grid_count = 0

                self.positives = 0
                self.negatives = 0
                self.mixed = 0

            def add_info(self, n_pos, n_neg, n_mixed):
                self.positives += n_pos
                self.negatives += n_neg
                self.mixed += n_mixed
                self.grid_count += 1

            def compile_report(self):
                report = f"matches    => {self.positives}\n"
                report += f"negatives  => {self.negatives}\n"
                report += f"mixed      => {self.mixed}\n"
                report += f"grid cells => {self.grid_count}\n"
                return report

        def __init__(self, filt, root_parent, *args, **kwargs):
            super(SpaceAnalyze.FilterApply, self).__init__(*args, **kwargs)
            self.filter = filt
            self.text = filt.tag

            self.alert = False
            self.interrupted = False

            self.root_parent = root_parent

            self.data = SpaceAnalyze.FilterApply.Analytics()

        def sample_report(self, img):
            predict = self.filter.predict(img)
            percent_predict = 100 * 2 * (predict - 0.5)

            if percent_predict[Filter.POSITIVE] > 50:
                report  = f"{self.filter.tag} +Positive {percent_predict[Filter.POSITIVE]:2.2f}%\n"
            
                self.data.add_info(1, 0, 0)
                self.interrupt()
                self.root_parent.ids.sub_sample.set_image(img)    
                return report, True
            else:
                report  = f"{self.filter.tag} -Negative {-percent_predict[Filter.NEGATIVE]:2.2f}%\n"
            
                self.data.add_info(0, 1, 0)

            return report, False

        def interrupt(self):
            self.ids.btn_train.text = 'Analyze [Next Cell]'
            self.interrupted = True
            self.root_parent.scan_interrupt()

        def analysis_callback(self):
            self.ids.btn_train.text = 'Analyzing ...'
            self.root_parent.ids.sub_sample.set_image(np.zeros((10, 10)))    
            
            if self.interrupted:
                self.root_parent.scan_continue()
            else:
                self.root_parent.scan_begin(self)

        def complete(self):
            self.root_parent.scan_reset()
            self.ids.btn_train.text = 'Analyze [DONE]'           
            self.root_parent.set_report(self.data.compile_report())

        def reset(self):
            self.root_parent.scan_reset()
            self.ids.btn_train.text = 'Analyze'

            self.data = SpaceAnalyze.FilterApply.Analytics()
            self.root_parent.scan_reset()

    def __init__(self, *args, **kwargs):
        super(SpaceAnalyze, self).__init__(**kwargs)
        self.loadFilters()
        self.scan_event = None
        self.scan_iter = None
        self.interrupted = False

    def loadFilters(self):
        self.ids.filter_list.clear_widgets()

        for i in Filter.loadAllFilters():
            widget = SpaceAnalyze.FilterApply(i, self)
            self.ids.filter_list.add_widget(widget)

    def contrast_img(self, img):
        max_val = np.amax(img)
        if max_val != 0:
            img = img.astype(float) * 255.0 / max_val
        
        img = img.astype(np.uint8)
        img = cv2.fastNlMeansDenoisingColored(img, None, 2, 10)

        img = img.astype('uint8')

        return img

    def find_scale(self, img):
        print("FINDING SCALE")
        
        w, h = img.shape[0:2]

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 50
        params.maxThreshold = 220

        params.filterByArea = True
        params.minArea = 200
        params.maxArea = w * h // 4

        params.filterByCircularity = True
        params.minCircularity = 0.2

        params.filterByConvexity = True
        params.minConvexity = 0.05

        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)

        diam = []
        for i in keypoints:
            diam.append(i.size)

        scl = int(np.average(diam)) if diam else min(img.shape[0:2])
        print("SCALE", scl)

        return scl

    def gridsplit_img(self, img, scale):
        print("GRIDSPLITTING")
        I, J = img.shape[:2]
        step = scale
        scan_list = []

        for i in range(0, I - scale, step):
            for j in range(0, J - scale, step):
                scan_list.append((i, j))
        for i in range(0, I - self.scale, step):
            scan_list.append((i, J - scale))
        for j in range(0, J - scale, step):
            scan_list.append((I - scale, j))
        
        scan_list.append((I - scale, J - scale))
        
        print("DONE")
        return scan_list, iter(scan_list)

    def scan_interrupt(self):
        self.interrupted = True    

    def scan_continue(self):
        if self.interrupted and self.scan_iter is not None:
            self.interrupted = False
            self.scan_event = Clock.schedule_interval(lambda t: self.scan_iterate(), 0)

    def scan_reset(self):
        self.set_report('REPORT')
        self.scan_interrupt()
        self.scan_event.cancel()
        self.scan_iter = None
        self.scan_list = []

    def set_report(self, text):
        self.report = text
        self.ids.info.text = text

    def scan_iterate(self):
        try:
            i, j = next(self.scan_iter)
        except StopIteration:
            self.current_filter.complete()
            self.scan_iter = None
            self.interrupted = True
        else:
            self.count += 1

            sub_sample = self.img[i:(i + self.scale), j:(j + self.scale)]
            self.ids.slide.update_viewbox(j, i, self.scale, self.scale)
            
            sub_report, alert = self.current_filter.sample_report(sub_sample)

            report = "-- ALERT --\n" if alert else "-- REPORT --\n"
            report += sub_report

            report += f"Scale     = {self.scale}\n" 
            report += f"Grid Cell - {self.count} of {len(self.scan_list)}\n"
            report += f"Position  - {(i, j)} in {self.img.shape[0:2]}\n"
            
            self.set_report(report)
                
            if alert:
                self.interrupted = True
            
        return not self.interrupted

    def scan_begin(self, filt):
        if self.scan_iter is None:
            self.count = 0

            self.current_filter = filt
            self.img = self.ids.slide.source_img
            self.scale = self.find_scale(self.contrast_img(self.img))
            self.scan_list, self.scan_iter = self.gridsplit_img(self.img, self.scale)
        
            self.interrupted = True
            self.scan_continue()

class FilterTrain(StackLayout):
    def __init__(self, filt, plot, *args, **kwargs):
        super(FilterTrain, self).__init__(*args, **kwargs)
        self.filter = filt
        self.text = filt.tag
        self.plot = plot

    def train(self):
        self.filter.train(self.plot)

    def save(self):
        self.filter.save()
        self.ids.btn_save.text = 'Saved'

class SpaceTrain(Screen):
    def load_filters(self):
        self.ids.filter_editor.clear_widgets()
    
        for tag in APL_Database.getAllTags():
            self.ids.filter_editor.add_widget(FilterTrain(Filter(tag), self.ids.plot))

class SpaceCategorize(Screen):
    def __init__(self, *args, **kwargs):
        super(SpaceCategorize, self).__init__(*args, **kwargs)
        self.scale = IMG_SCALE

        self.i, self.j = 0, 0

    def load_slide(self):
        self.ids.slide.popup_selectImage(self.next_sample)

    def next_sample(self):
        try:
            slide = self.ids.slide.source_img

            I, J = np.size(slide, 0), np.size(slide, 1)

            subsample = slide[self.i:(self.i + self.scale ), self.j:(self.j + self.scale)]

            self.ids.sub_sample.set_image(subsample)
            self.ids.slide.update_viewbox(self.j, self.i, self.scale, self.scale)
        
            if self.i < I - 2 * self.scale:
                self.i += self.scale // 2
            else:
                self.i = 0
                
                if self.j < J - 2 * self.scale:
                    self.j += self.scale // 2
                else:
                    self.j = 0
                
        except Exception as ex:
            print(ex)

    def save_tags(self):
        img = self.ids.sub_sample.source_img

        tags = []
        for i in self.ids.tags.children:
            if isinstance(i, ChipRemovable):
                if i.selected:
                    tags.append(i.text + '+ve')
                else:
                    tags.append(i.text + '-ve')

        APL_Database.saveImage(img, tags)

    def add_tag(self, tag):
        if tag:
            chip = ChipRemovable()
            chip = ChipRemovable()

            chip.text = tag
            chip.remove = lambda: self.ids.tags.remove_widget(chip)
            
            self.ids.tags.add_widget(chip)

class SpaceInterfaceOverview(Screen): pass
class SpaceCredits(Screen): pass

class WorkSpace(ScreenManager):
    def __init__(self, **kwargs):
        super(WorkSpace, self).__init__(**kwargs)

        self.add_widget(SpaceStart(name = 'screen_start'))
        self.add_widget(SpaceAnalyze(name = 'screen_analyze'))
        self.add_widget(SpaceCreateSlide(name = 'screen_createslide'))
        self.add_widget(SpaceTrain(name = 'screen_train'))
        self.add_widget(SpaceCategorize(name = 'screen_categorize'))
        self.add_widget(SpaceCredits(name = 'screen_credits'))

class MainWindow(BoxLayout): pass
class Application(App): 
    def build(self):
        self.title = 'MicroLab - DeepStain'
        return MainWindow()

Builder.load_file('src/style.kv')

if __name__ == '__main__':
    try:
        Application().run()
    except Exception as ex:
        print("Error:", ex)