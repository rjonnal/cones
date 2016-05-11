import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import numpy as np
import cones

DEFAULT_SIGMA = 2.0
DEFAULT_FRAC = 0.001
DEFAULT_NBINS = 50
DEFAULT_DIAMETER = 5
DEFAULT_NEIGHBORHOOD = 15

WIDGET_HEIGHT = 20

class AppForm(QMainWindow):
    def __init__(self, parent=None,dataset=None,info=''):
        QMainWindow.__init__(self, parent)
        self.dataset = dataset
        self.setWindowTitle(dataset.tag+':'+info)
        self.create_main_frame()
        self.set_load_button()
        self.display_image_1 = self.dataset.im0
        self.display_image_2 = self.dataset.gbs
        self.on_draw()
        
    def save_plot(self):
        file_choices = "PNG (*.png)|*.png"
        
        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)

    def set_load_button(self):
        self.load_button.setEnabled(self.dataset.has_coordinates_file())

    def on_click(self,event):
        xclick,yclick = event.xdata,event.ydata

        #label = self.dataset.get_label(xclick,yclick)

        if event.button==1:
            label = 's'
        else:
            label = 'l/m'
        
        self.dataset.coneset.edit(xclick,yclick,label)

        self.on_draw()


    def set_red(self):
        self.display_image_2 = self.dataset.rbs
        self.on_draw()
    
    def set_green(self):
        self.display_image_2 = self.dataset.gbs
        self.on_draw()

    def save_coordinates(self):
        print 'save...',
        fn = self.dataset.coordinate_fn
        self.dataset.coneset.to_file(fn)
        self.dataset.imshow(fn=self.dataset.marked_fn)
        self.set_load_button()
        print 'done'

    def load_coordinates(self):
        print 'load...',
        fn = self.dataset.coordinate_fn
        self.dataset.coneset.from_file(fn)
        self.on_draw()
        print 'done'
        
    def on_draw(self):
        """ Redraws the figure
        """
        self.axes_1.clear()
        self.axes_1.imshow(self.display_image_1)
        self.axes_1.autoscale(False)
        self.dataset.coneset.plot_on(self.axes_1)
        self.axes_2.clear()
        imh = self.axes_2.imshow(self.display_image_2)
        self.axes_2.autoscale(False)
        imh.set_cmap('gray')
        self.dataset.coneset.plot_on(self.axes_2)
        self.canvas.draw()

    def create_panel(self,name,parameter_names,function,defaults=None,dtypes=None):
        if defaults is None:
            defaults = [0.0]*len(parameter_names)
        if dtypes is None:
            dtypes = [float]*len(parameter_names)
            
        layout = QHBoxLayout()
        button = QPushButton(name)
        button.setFixedSize(100,WIDGET_HEIGHT)
        boxes = []
        for parameter_name,default in zip(parameter_names,defaults):
            qle = QLineEdit()
            try:
                qle.setText('%0.3f'%default)
            except TypeError as te:
                qle.setText(default)
            boxes.append(qle)
            
        def f():
            params = []
            for box,dtype in zip(boxes,dtypes):
                print box.text(),dtype
                params.append(dtype(unicode(box.text())))
            function(*params)
            self.on_draw()
        self.connect(button,SIGNAL('clicked ()'),f)

        layout.addWidget(button)
        for box in boxes:
            layout.addWidget(box)

        return layout
        
    def create_main_frame(self):
        self.main_frame = QWidget()
        
        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        self.fig = Figure((10.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        
        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        #self.axes_1 = self.fig.add_subplot(121)
        self.axes_1 = self.fig.add_axes([0,0,.5,1.0])
        self.axes_2 = self.fig.add_axes([.5,0,.5,1.0])
        
        # Bind the 'pick' event for clicking on one of the bars
        #
        self.canvas.mpl_connect('button_release_event', self.on_click)
        
        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
        self.load_button = QPushButton('&Load')
        self.connect(self.load_button, SIGNAL('clicked ()'), self.load_coordinates)

        self.save_button = QPushButton('&Save')
        self.connect(self.save_button, SIGNAL('clicked ()'), self.save_coordinates)

        self.red_button = QPushButton('&Red')
        self.connect(self.red_button, SIGNAL('clicked ()'), self.set_red)

        self.green_button = QPushButton('&Green')
        self.connect(self.green_button, SIGNAL('clicked ()'), self.set_green)
        
        self.quit_button = QPushButton('&Quit')
        self.connect(self.quit_button, SIGNAL('clicked ()'), sys.exit)
        
        #
        # Layout with box sizers
        # 
        load_save_panel = QHBoxLayout()
        
        load_save_panel.addWidget(self.load_button)
        load_save_panel.addWidget(self.save_button)
        load_save_panel.addWidget(self.red_button)
        load_save_panel.addWidget(self.green_button)
        load_save_panel.addWidget(self.quit_button)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(load_save_panel)

        #print_panel = self.create_panel('print',['dpi','filename'],self.dataset.imshow,defaults=[50,self.dataset.marked_fn],dtypes=[lambda x: int(round(float(x))),str])
        #vbox.addLayout(print_panel)
        autodetect_panel = self.create_panel('peaks',['sigma','frac','nbins','diameter'],self.dataset.autodetect_peaks,defaults=[2.0,.001,50,5],dtypes=[float,float,lambda x: int(round(float(x))),lambda x: int(round(float(x)))])
        vbox.addLayout(autodetect_panel)
        autodetect_alt_panel = self.create_panel('alt_peaks',['sigma','neighborhood'],self.dataset.autodetect_peaks_alt,defaults=[1.5,15],dtypes=[float,lambda x: int(round(float(x)))])
        vbox.addLayout(autodetect_alt_panel)
        clear_panel = self.create_panel('clear',['filter'],self.dataset.coneset.clear,defaults=['l/m'],dtypes=[str])
        vbox.addLayout(clear_panel)
        clear_below_panel = self.create_panel('clear below',['frac','filt','rad'],self.dataset.clear_below,defaults=[0.2,'l/ms',2],dtypes=[float,str,lambda x: int(round(float(x)))])
        vbox.addLayout(clear_below_panel)
        
        
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
        self.showMaximized()
    
    # def add_actions(self, target, actions):
    #     for action in actions:
    #         if action is None:
    #             target.addSeparator()
    #         else:
    #             target.addAction(action)

    # def create_action(  self, text, slot=None, shortcut=None, 
    #                     icon=None, tip=None, checkable=False, 
    #                     signal="triggered()"):
    #     action = QAction(text, self)
    #     if icon is not None:
    #         action.setIcon(QIcon(":/%s.png" % icon))
    #     if shortcut is not None:
    #         action.setShortcut(shortcut)
    #     if tip is not None:
    #         action.setToolTip(tip)
    #         action.setStatusTip(tip)
    #     if slot is not None:
    #         self.connect(action, SIGNAL(signal), slot)
    #     if checkable:
    #         action.setCheckable(True)
    #     return action


def main():
    app = QApplication(sys.argv)
    form = AppForm(filename=sys.argv[1])
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
