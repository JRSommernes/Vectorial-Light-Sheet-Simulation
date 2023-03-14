from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from microscope import Microscope
import tifffile, sys, os, json
import numpy as np

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

class PlotCanvas(FigureCanvasQTAgg):
    """
    This class is a subclass of FigureCanvasQTAgg, which is a Qt widget for displaying a
    Matplotlib figure. It is used to display a graphical representation of a lens system.

    Parameters:
    system (object): The lens system to be plotted
    parent (QtWidgets.QWidget, optional): The parent widget for the canvas
    width (float, optional): Width of the figure in inches
    height (float, optional): Height of the figure in inches
    dpi (int, optional): Dots per inch for the figure
    """
    def __init__(self, system, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.system = system

        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.plot()

    def plot(self):
        """
        This method is used to create an empty plot with no ticks on x and y axis
        """
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw()

    def update_plot(self):
        """
        This method is used to update the plot with the lens system data
        """
        focus_x = [0]
        focus_y = [0]
        lenses_x = []
        lenses_y = []
        rot = 0
        modes = ['col','foc']
        for i,lens in enumerate(self.system.lenses):
            mode = modes[i%2]
            if mode == 'col' and i != 0:
                rot += lens.rot

            tmp_x = np.cos(rot)/lens.NA
            tmp_y = np.sin(rot)/lens.NA

            lenses_x.append(focus_x[i]+tmp_x)
            lenses_y.append(focus_y[i]+tmp_y)
            focus_x.append(lenses_x[i]+tmp_x)
            focus_y.append(lenses_y[i]+tmp_y)

            if mode == 'foc':
                rot += lens.rot

        if len(self.system.lenses) > 0:
            lenses_y /= focus_x[-1]
            lenses_x /= focus_x[-1]
            focus_y /= focus_x[-1]
            focus_x /= focus_x[-1]

        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        rot = np.pi/2
        for i,lens in enumerate(self.system.lenses):
            if i % 2 == 0:
                tmp_rot = lens.rot
                rot += tmp_rot
            center = np.array((lenses_x[i], lenses_y[i]))
            top = 0.1*np.array((np.cos(rot),np.sin(rot)))+center
            bottom = -0.1*np.array((np.cos(rot),np.sin(rot)))+center
            self.ax.annotate("",xy=top, xycoords='data', xytext=bottom, textcoords='data',
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='r', lw=2))

            if i % 2 != 0:
                tmp_rot = lens.rot
                rot += tmp_rot

        if hasattr(self.system, 'camera'):
            center = np.array((focus_x[-1], focus_y[-1]))
            top = 0.1*np.array((np.cos(rot),np.sin(rot)))+center
            bottom = -0.1*np.array((np.cos(rot),np.sin(rot)))+center
            self.ax.annotate("",xy=top, xycoords='data', xytext=bottom, textcoords='data',
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color='r', lw=2))

        self.ax.plot(focus_x,focus_y,'-.',c='b')
        self.ax.axis('equal')
        self.draw()


class Ui_MainWindow(object):
    def setupUi(self,MainWindow):
        self.title = 'Optics simulations'
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 400

        ########################################################################
        #Initialize

        self.add_system()
        MainWindow.setObjectName(self.title)
        MainWindow.resize(self.width, self.height)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)

        ########################################################################
        #Widget creation

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, self.height//2, self.width, self.height//2))
        self.tabWidget.setObjectName("tabWidget")

        ########################################################################
        #Lens creation tab

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.NA_lab = QtWidgets.QLabel(self.tab)
        self.NA_lab.setGeometry(QtCore.QRect(10, 10, 85, 20))
        self.NA_lab.setObjectName("NA_lab")

        self.NA_lineEdit = QtWidgets.QLineEdit(self.tab)
        self.NA_lineEdit.setGeometry(QtCore.QRect(100, 10, 50, 20))
        self.NA_lineEdit.setObjectName("NA_lineEdit")

        self.HelpNA = QtWidgets.QPushButton(self.tab)
        self.HelpNA.setGeometry(QtCore.QRect(160, 10, 50, 20))
        self.HelpNA.setObjectName("HelpNA")

        self.RI_lab = QtWidgets.QLabel(self.tab)
        self.RI_lab.setGeometry(QtCore.QRect(10, 32, 85, 20))
        self.RI_lab.setObjectName("RI_lab")

        self.RI_lineEdit = QtWidgets.QLineEdit(self.tab)
        self.RI_lineEdit.setGeometry(QtCore.QRect(100, 32, 50, 20))
        self.RI_lineEdit.setObjectName("RI_lineEdit")

        self.HelpRI = QtWidgets.QPushButton(self.tab)
        self.HelpRI.setGeometry(QtCore.QRect(160, 32, 50, 20))
        self.HelpRI.setObjectName("HelpRI")

        self.rot_lab = QtWidgets.QLabel(self.tab)
        self.rot_lab.setGeometry(QtCore.QRect(10, 54, 85, 20))
        self.rot_lab.setObjectName("rot_lab")

        self.rot_lineEdit = QtWidgets.QLineEdit(self.tab)
        self.rot_lineEdit.setGeometry(QtCore.QRect(100, 54, 50, 20))
        self.rot_lineEdit.setObjectName("rot_lineEdit")

        self.Helprot = QtWidgets.QPushButton(self.tab)
        self.Helprot.setGeometry(QtCore.QRect(160, 54, 50, 20))
        self.Helprot.setObjectName("Helprot")

        self.pos_lab = QtWidgets.QLabel(self.tab)
        self.pos_lab.setGeometry(QtCore.QRect(10, 76, 85, 20))
        self.pos_lab.setObjectName("pos_lab")

        self.pos_lineEdit = QtWidgets.QLineEdit(self.tab)
        self.pos_lineEdit.setGeometry(QtCore.QRect(100, 76, 50, 20))
        self.pos_lineEdit.setObjectName("pos_lineEdit")

        self.Helppos = QtWidgets.QPushButton(self.tab)
        self.Helppos.setGeometry(QtCore.QRect(160, 76, 50, 20))
        self.Helppos.setObjectName("Helppos")

        self.AddLens = QtWidgets.QPushButton(self.tab)
        self.AddLens.setGeometry(QtCore.QRect(10, 105, 75, 25))
        self.AddLens.setObjectName("AddLens")

        ########################################################################
        #Camera creation tab

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        self.pixels_lab = QtWidgets.QLabel(self.tab_2)
        self.pixels_lab.setGeometry(QtCore.QRect(10, 10, 85, 20))
        self.pixels_lab.setObjectName("pixels_lab")

        self.pixels_lineEdit = QtWidgets.QLineEdit(self.tab_2)
        self.pixels_lineEdit.setGeometry(QtCore.QRect(100, 10, 50, 20))
        self.pixels_lineEdit.setObjectName("pixels_lineEdit")

        self.Helppixel = QtWidgets.QPushButton(self.tab_2)
        self.Helppixel.setGeometry(QtCore.QRect(160, 10, 50, 20))
        self.Helppixel.setObjectName("Helppixel")

        self.voxel_lab = QtWidgets.QLabel(self.tab_2)
        self.voxel_lab.setGeometry(QtCore.QRect(10, 32, 85, 20))
        self.voxel_lab.setObjectName("voxel_lab")

        self.voxel_lineEdit = QtWidgets.QLineEdit(self.tab_2)
        self.voxel_lineEdit.setGeometry(QtCore.QRect(100, 32, 50, 20))
        self.voxel_lineEdit.setObjectName("voxel_lineEdit")

        self.Helpvoxel = QtWidgets.QPushButton(self.tab_2)
        self.Helpvoxel.setGeometry(QtCore.QRect(160, 32, 50, 20))
        self.Helpvoxel.setObjectName("Helpvoxel")

        self.RMS_lab = QtWidgets.QLabel(self.tab_2)
        self.RMS_lab.setGeometry(QtCore.QRect(10, 54, 85, 20))
        self.RMS_lab.setObjectName("RMS_lab")

        self.RMS_lineEdit = QtWidgets.QLineEdit(self.tab_2)
        self.RMS_lineEdit.setGeometry(QtCore.QRect(100, 54, 50, 20))
        self.RMS_lineEdit.setObjectName("RMS_lineEdit")

        self.HelpRMS = QtWidgets.QPushButton(self.tab_2)
        self.HelpRMS.setGeometry(QtCore.QRect(160, 54, 50, 20))
        self.HelpRMS.setObjectName("HelpRMS")

        self.offset_lab = QtWidgets.QLabel(self.tab_2)
        self.offset_lab.setGeometry(QtCore.QRect(10, 76, 85, 20))
        self.offset_lab.setObjectName("offset_lab")

        self.offset_lineEdit = QtWidgets.QLineEdit(self.tab_2)
        self.offset_lineEdit.setGeometry(QtCore.QRect(100, 76, 50, 20))
        self.offset_lineEdit.setObjectName("offset_lineEdit")

        self.Helpoffset = QtWidgets.QPushButton(self.tab_2)
        self.Helpoffset.setGeometry(QtCore.QRect(160, 76, 50, 20))
        self.Helpoffset.setObjectName("Helpoffset")

        self.OTF_size_lab = QtWidgets.QLabel(self.tab_2)
        self.OTF_size_lab.setGeometry(QtCore.QRect(310, 10, 85, 20))
        self.OTF_size_lab.setObjectName("OTF_size_lab")

        self.OTF_size_lineEdit = QtWidgets.QLineEdit(self.tab_2)
        self.OTF_size_lineEdit.setGeometry(QtCore.QRect(380, 10, 50, 20))
        self.OTF_size_lineEdit.setObjectName("OTF_size_lineEdit")

        self.HelpOTF = QtWidgets.QPushButton(self.tab_2)
        self.HelpOTF.setGeometry(QtCore.QRect(440, 10, 50, 20))
        self.HelpOTF.setObjectName("HelpOTF")

        self.FoV_lab = QtWidgets.QLabel(self.tab_2)
        self.FoV_lab.setGeometry(QtCore.QRect(310, 32, 85, 20))
        self.FoV_lab.setObjectName("FoV_lab")

        self.FoV_lab_2 = QtWidgets.QLabel(self.tab_2)
        self.FoV_lab_2.setGeometry(QtCore.QRect(400, 32, 100, 20))
        self.FoV_lab_2.setObjectName("FoV_lab_2")

        self.AddCamera = QtWidgets.QPushButton(self.tab_2)
        self.AddCamera.setGeometry(QtCore.QRect(10, 105, 75, 25))
        self.AddCamera.setObjectName("AddCamera")

        ########################################################################
        #Light-sheet tab
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")

        self.ls_opening_lab = QtWidgets.QLabel(self.tab_3)
        self.ls_opening_lab.setGeometry(QtCore.QRect(10, 10, 85, 20))
        self.ls_opening_lab.setObjectName("ls_opening_lab")

        self.ls_opening_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.ls_opening_lineEdit.setGeometry(QtCore.QRect(100, 10, 50, 20))
        self.ls_opening_lineEdit.setObjectName("ls_opening_lineEdit")

        self.Helpopening = QtWidgets.QPushButton(self.tab_3)
        self.Helpopening.setGeometry(QtCore.QRect(160, 10, 50, 20))
        self.Helpopening.setObjectName("Helpopening")

        self.lam_ex_lab = QtWidgets.QLabel(self.tab_3)
        self.lam_ex_lab.setGeometry(QtCore.QRect(10, 32, 85, 20))
        self.lam_ex_lab.setObjectName("lam_ex_lab")

        self.ex_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.ex_lineEdit.setGeometry(QtCore.QRect(100, 32, 50, 20))
        self.ex_lineEdit.setObjectName("ex_lineEdit")

        self.Helplamex = QtWidgets.QPushButton(self.tab_3)
        self.Helplamex.setGeometry(QtCore.QRect(160, 32, 50, 20))
        self.Helplamex.setObjectName("Helplamex")

        self.pol_ex_lab = QtWidgets.QLabel(self.tab_3)
        self.pol_ex_lab.setGeometry(QtCore.QRect(10, 54, 85, 20))
        self.pol_ex_lab.setObjectName("pol_ex_lab")

        self.pol_ex_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.pol_ex_lineEdit.setGeometry(QtCore.QRect(100, 54, 50, 20))
        self.pol_ex_lineEdit.setObjectName("pol_ex_lineEdit")

        self.Helppolex = QtWidgets.QPushButton(self.tab_3)
        self.Helppolex.setGeometry(QtCore.QRect(160, 54, 50, 20))
        self.Helppolex.setObjectName("Helppolex")

        self.ls_thic_lab = QtWidgets.QLabel(self.tab_3)
        self.ls_thic_lab.setGeometry(QtCore.QRect(10, 76, 85, 20))
        self.ls_thic_lab.setObjectName("ls_thic_lab")

        self.ls_thic_lab_2 = QtWidgets.QLabel(self.tab_3)
        self.ls_thic_lab_2.setGeometry(QtCore.QRect(100, 76, 50, 20))
        self.ls_thic_lab_2.setObjectName("ls_thic_lab_2")

        self.makeLS = QtWidgets.QPushButton(self.tab_3)
        self.makeLS.setGeometry(QtCore.QRect(10, 105, 75, 25))
        self.makeLS.setObjectName("makeLS")

        ########################################################################
        #Tracing tab

        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")

        self.ensamble_lab = QtWidgets.QLabel(self.tab_4)
        self.ensamble_lab.setGeometry(QtCore.QRect(10, 10, 85, 20))
        self.ensamble_lab.setObjectName("ensamble_lab")

        self.ensamble_lineEdit = QtWidgets.QLineEdit(self.tab_4)
        self.ensamble_lineEdit.setGeometry(QtCore.QRect(100, 10, 50, 20))
        self.ensamble_lineEdit.setObjectName("ensamble_lineEdit")

        self.Helpensamble = QtWidgets.QPushButton(self.tab_4)
        self.Helpensamble.setGeometry(QtCore.QRect(160, 10, 50, 20))
        self.Helpensamble.setObjectName("Helpensamble")

        self.lam_em_lab = QtWidgets.QLabel(self.tab_4)
        self.lam_em_lab.setGeometry(QtCore.QRect(10, 32, 85, 20))
        self.lam_em_lab.setObjectName("lam_em_lab")

        self.em_lineEdit = QtWidgets.QLineEdit(self.tab_4)
        self.em_lineEdit.setGeometry(QtCore.QRect(100, 32, 50, 20))
        self.em_lineEdit.setObjectName("em_lineEdit")

        self.Helplamem = QtWidgets.QPushButton(self.tab_4)
        self.Helplamem.setGeometry(QtCore.QRect(160, 32, 50, 20))
        self.Helplamem.setObjectName("Helplamem")

        self.SNR_lab = QtWidgets.QLabel(self.tab_4)
        self.SNR_lab.setGeometry(QtCore.QRect(10, 54, 85, 20))
        self.SNR_lab.setObjectName("SNR_lab")

        self.SNR_lineEdit = QtWidgets.QLineEdit(self.tab_4)
        self.SNR_lineEdit.setGeometry(QtCore.QRect(100, 54, 50, 20))
        self.SNR_lineEdit.setObjectName("SNR_lineEdit")

        self.HelpSNR = QtWidgets.QPushButton(self.tab_4)
        self.HelpSNR.setGeometry(QtCore.QRect(160, 54, 50, 20))
        self.HelpSNR.setObjectName("HelpSNR")

        self.ani_lab = QtWidgets.QLabel(self.tab_4)
        self.ani_lab.setGeometry(QtCore.QRect(10, 76, 85, 20))
        self.ani_lab.setObjectName("ani_lab")

        self.checkBox_ani = QtWidgets.QCheckBox(self.tab_4)
        self.checkBox_ani.setGeometry(QtCore.QRect(100, 76, 50, 20))
        self.checkBox_ani.setObjectName("checkBox_ani")

        self.Helpani = QtWidgets.QPushButton(self.tab_4)
        self.Helpani.setGeometry(QtCore.QRect(160, 76, 50, 20))
        self.Helpani.setObjectName("Helpani")

        self.mag_lab = QtWidgets.QLabel(self.tab_4)
        self.mag_lab.setGeometry(QtCore.QRect(310, 10, 60, 20))
        self.mag_lab.setObjectName("mag_lab")

        self.mag_lab_2 = QtWidgets.QLabel(self.tab_4)
        self.mag_lab_2.setGeometry(QtCore.QRect(380, 10, 50, 20))
        self.mag_lab_2.setObjectName("mag_lab_2")

        self.axmag_lab = QtWidgets.QLabel(self.tab_4)
        self.axmag_lab.setGeometry(QtCore.QRect(310, 32, 60, 20))
        self.axmag_lab.setObjectName("axmag_lab")

        self.axmag_lab_2 = QtWidgets.QLabel(self.tab_4)
        self.axmag_lab_2.setGeometry(QtCore.QRect(380, 32, 50, 20))
        self.axmag_lab_2.setObjectName("axmag_lab_2")

        self.savename_lab = QtWidgets.QLabel(self.tab_4)
        self.savename_lab.setGeometry(QtCore.QRect(310, 54, 60, 20))
        self.savename_lab.setObjectName("savename_lab")

        self.savename_lineEdit = QtWidgets.QLineEdit(self.tab_4)
        self.savename_lineEdit.setGeometry(QtCore.QRect(380, 54, 150, 20))
        self.savename_lineEdit.setObjectName("savename_lineEdit")

        self.TraceSystem = QtWidgets.QPushButton(self.tab_4)
        self.TraceSystem.setGeometry(QtCore.QRect(310, 105, 60, 25))
        self.TraceSystem.setObjectName("TraceSystem")

        self.pbar = QtWidgets.QProgressBar(self.tab_4)
        self.pbar.setGeometry(380, 105, 185, 25)

        ########################################################################
        #Tab creation

        self.tabWidget.addTab(self.tab, "")
        self.tabWidget.addTab(self.tab_2, "")
        self.tabWidget.addTab(self.tab_3, "")
        self.tabWidget.addTab(self.tab_4, "")

        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.initPlot(MainWindow)

        ########################################################################
        #Button connection

        self.AddLens.clicked.connect(self.makeLens)
        self.AddCamera.clicked.connect(self.makeCamera)
        self.TraceSystem.clicked.connect(self.trace)
        self.makeLS.clicked.connect(self.make_light_sheet)
        self.HelpNA.clicked.connect(lambda: self.helpLens('NA'))
        self.HelpRI.clicked.connect(lambda: self.helpLens('RI'))
        self.Helprot.clicked.connect(lambda: self.helpLens('rot'))
        self.Helppos.clicked.connect(lambda: self.helpLens('pos'))
        self.Helppixel.clicked.connect(lambda: self.helpCamera('pixel'))
        self.Helpvoxel.clicked.connect(lambda: self.helpCamera('vox'))
        self.HelpRMS.clicked.connect(lambda: self.helpCamera('RMS'))
        self.Helpoffset.clicked.connect(lambda: self.helpCamera('offset'))
        self.HelpOTF.clicked.connect(lambda: self.helpCamera('OTF'))
        self.Helpopening.clicked.connect(lambda: self.helpLightSheet('opening'))
        self.Helplamex.clicked.connect(lambda: self.helpLightSheet('lam'))
        self.Helppolex.clicked.connect(lambda: self.helpLightSheet('pol'))
        self.Helpensamble.clicked.connect(lambda: self.helpTracing('ensamble'))
        self.Helplamem.clicked.connect(lambda: self.helpTracing('lam'))
        self.HelpSNR.clicked.connect(lambda: self.helpTracing('SNR'))
        self.Helpani.clicked.connect(lambda: self.helpTracing('ani'))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        ########################################################################
        #Tabs

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Lenses"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Camera"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Light-sheet"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Tracing"))

        ########################################################################
        #Lens tab

        self.NA_lab.setText(_translate("MainWindow", "Lens NA: "))
        self.HelpNA.setText(_translate("MainWindow", "Help"))

        self.RI_lab.setText(_translate("MainWindow", "Immersion RI: "))
        self.RI_lineEdit.setText(_translate("MainWindow", "1"))
        self.HelpRI.setText(_translate("MainWindow", "Help"))

        self.rot_lab.setText(_translate("MainWindow", "Lens rotation [°]: "))
        self.rot_lineEdit.setText(_translate("MainWindow", "0"))
        self.Helprot.setText(_translate("MainWindow", "Help"))

        self.pos_lab.setText(_translate("MainWindow", "Lens position: "))
        self.pos_lineEdit.setText(_translate("MainWindow", "0"))
        self.Helppos.setText(_translate("MainWindow", "Help"))

        self.AddLens.setText(_translate("MainWindow", "Add Lens"))

        ########################################################################
        #Camera tab

        self.pixels_lab.setText(_translate("MainWindow", "Number of pixels:"))
        self.pixels_lineEdit.setText(_translate("MainWindow", "128"))
        self.Helppixel.setText(_translate("MainWindow", "Help"))

        self.voxel_lab.setText(_translate("MainWindow", "Voxel size [um]:"))
        self.voxel_lineEdit.setText(_translate("MainWindow", "2"))
        self.Helpvoxel.setText(_translate("MainWindow", "Help"))

        self.RMS_lab.setText(_translate("MainWindow", "Readout noise:"))
        self.RMS_lineEdit.setText(_translate("MainWindow", "1.4"))
        self.HelpRMS.setText(_translate("MainWindow", "Help"))

        self.offset_lab.setText(_translate("MainWindow", "Base level:"))
        self.offset_lineEdit.setText(_translate("MainWindow", "100"))
        self.Helpoffset.setText(_translate("MainWindow", "Help"))

        self.OTF_size_lab.setText(_translate("MainWindow", "OTF size:"))
        self.OTF_size_lineEdit.setText(_translate("MainWindow", "256"))
        self.HelpOTF.setText(_translate("MainWindow", "Help"))

        self.FoV_lab.setText(_translate("MainWindow", "FoV sample [um]:"))
        self.FoV_lab_2.setText(_translate("MainWindow", "N/A"))

        self.AddCamera.setText(_translate("MainWindow", "Update"))

        ########################################################################
        #Light-sheet tab

        self.ls_opening_lab.setText(_translate("MainWindow", "LS opening [°]:"))
        self.ls_opening_lineEdit.setText(_translate("MainWindow", "5"))
        self.Helpopening.setText(_translate("MainWindow", "Help"))

        self.lam_ex_lab.setText(_translate("MainWindow", "λ exitation [nm]:"))
        self.ex_lineEdit.setText(str(round(self.system.lam_ex*1e9)))
        self.Helplamex.setText(_translate("MainWindow", "Help"))

        self.pol_ex_lab.setText(_translate("MainWindow", "exitation pol:"))
        self.pol_ex_lineEdit.setText(_translate("MainWindow", "p"))
        self.Helppolex.setText(_translate("MainWindow", "Help"))

        self.ls_thic_lab.setText(_translate("MainWindow", "LS length:"))
        self.ls_thic_lab_2.setText(_translate("MainWindow", "N/A"))

        self.makeLS.setText(_translate("MainWindow", "Update"))


        ########################################################################
        #Tracing tab

        self.ensamble_lab.setText(_translate("MainWindow", "Orientations:"))
        self.ensamble_lineEdit.setText('10')
        self.Helpensamble.setText(_translate("MainWindow", "Help"))

        self.lam_em_lab.setText(_translate("MainWindow", "λ emission [nm]:"))
        self.em_lineEdit.setText(str(round(self.system.lam_em*1e9)))
        self.Helplamem.setText(_translate("MainWindow", "Help"))

        self.SNR_lab.setText(_translate("MainWindow", "SNR:"))
        self.SNR_lineEdit.setText('100')
        self.HelpSNR.setText(_translate("MainWindow", "Help"))

        self.ani_lab.setText(_translate("MainWindow", "Anisotropy:"))
        self.Helpani.setText(_translate("MainWindow", "Help"))

        self.mag_lab.setText(_translate("MainWindow", "Lateral mag: "))
        self.mag_lab_2.setText(_translate("MainWindow", "N/A"))

        self.axmag_lab.setText(_translate("MainWindow", "Axial mag: "))
        self.axmag_lab_2.setText(_translate("MainWindow", "N/A"))

        self.savename_lab.setText(_translate("MainWindow", "Save name:"))
        self.savename_lineEdit.setText('tmp')

        self.TraceSystem.setText(_translate("MainWindow", "Trace"))

    def initPlot(self,MainWindow):
        self.figure = PlotCanvas(self.system, MainWindow, width=self.width//100, height=self.height//200)
        self.figure.move(0,0)
        MainWindow.show()

    def add_system(self):
        lam_ex = 488e-9
        lam_em = 507e-9
        path = 'tmp'
        self.system = Microscope(lam_ex,lam_em,path)

    def makeLens(self):
        try:
            NA = float(self.NA_lineEdit.text())
        except:
            self.critError('NA')
            return
        try:
            RI = float(self.RI_lineEdit.text())
        except:
            self.critError('RI')
            return
        try:
            rot = float(self.rot_lineEdit.text())*np.pi/180
        except:
            self.critError('rot')
            return
        try:
            pos = int(self.pos_lineEdit.text())
        except:
            self.critError('pos')
            return

        self.system.add_lens(NA,RI,rot,pos)
        self.figure.update_plot()
        self.update_system()

        self.NA_lineEdit.setText('')
        self.RI_lineEdit.setText('1')
        self.rot_lineEdit.setText('0')
        self.pos_lineEdit.setText('0')

    def makeCamera(self):
        try:
            res = int(self.pixels_lineEdit.text())
        except:
            self.critError('resolution')
            return
        try:
            vox = float(self.voxel_lineEdit.text())*1e-6
        except:
            self.critError('voxel size')
            return
        try:
            RMS = float(self.RMS_lineEdit.text())
        except:
            self.critError('RMS')
            return
        try:
            offset = int(self.offset_lineEdit.text())
        except:
            self.critError('offset')
            return
        try:
            self.system.OTF_res = int(self.OTF_size_lineEdit.text())
        except:
            self.critError('OTF size')
            return

        self.system.add_camera(res,vox,offset,RMS)
        self.figure.update_plot()
        self.update_system()

    def make_light_sheet(self):
        try:
            self.system.ls_opening = float(self.ls_opening_lineEdit.text())*np.pi/180
        except:
            self.critError('lightsheet opening')
            return
        try:
            self.system.lam_ex = float(self.ex_lineEdit.text())*1e-9
        except:
            self.critError('lightsheet excitation')
            return

        valid_pol = ['p','s','u']
        pol_ex = self.pol_ex_lineEdit.text()
        if pol_ex in valid_pol:
            self.system.ls_pol = pol_ex
        else:
            self.critError('lightsheet polarization')
            return

        self.update_system()

    def update_system(self):
        num_lenses = len(self.system.lenses)
        if not hasattr(self.system, 'camera'):
            self.FoV_lab_2.setText('N/A')
            self.mag_lab_2.setText('N/A')
            self.axmag_lab_2.setText('N/A')
        elif num_lenses == 0:
            self.FoV_lab_2.setText('N/A')
            self.mag_lab_2.setText('N/A')
            self.axmag_lab_2.setText('N/A')
        elif num_lenses % 2 != 0:
            self.FoV_lab_2.setText('N/A')
            self.mag_lab_2.setText('N/A')
            self.axmag_lab_2.setText('N/A')
        else:
            self.system.calculate_system_specs()
            self.FoV_lab_2.setText(str(round(self.system.FoV*1e6,2)))
            self.mag_lab_2.setText(str(round(self.system.mag,2)))
            self.axmag_lab_2.setText(str(round(self.system.axial_mag,2)))


        try:
            RI = self.system.lenses[0].RI
            length = 1e6*self.system.lam_ex/(RI*(1-np.cos(self.system.ls_opening)))
            self.ls_thic_lab_2.setText(str(round(length,1)))
        except:
            self.ls_thic_lab_2.setText('N/A')

    def trace(self):
        try:
            self.system.ensamble = int(self.ensamble_lineEdit.text())
        except:
            self.critError('ensamble')
            return

        try:
            self.system.lam_em = float(self.em_lineEdit.text())*1e-9
        except:
            self.critError('emission wavelength')
            return

        try:
            self.system.SNR = float(self.SNR_lineEdit.text())
        except:
            self.critError('Signal to noise ratio')
            return

        if self.checkBox_ani.isChecked():
            self.system.anisotropy = 0.4
        else:
            self.system.anisotropy = 0

        try:
            saveloc = self.savename_lineEdit.text()
        except:
            self.critError('Save name')
            return

        if not os.path.exists(saveloc):
            os.mkdir(saveloc)

        self.TraceSystem.setText('Working')

        self.system.calculate_PSF(self)

        metadata = {'Dipoles in ensamble' : self.system.ensamble,
                    'Emission wavelength [nm]' : np.round(self.system.lam_em*1e9,2),
                    'Excitation wavelength [nm]' : np.round(self.system.lam_ex*1e9,2),
                    'Full FoV [pixels]' : self.system.camera.res,
                    'Full FoV in object space [microns]' : self.system.FoV*1e6,
                    'Light sheet opening [degrees]' : np.round(self.system.ls_opening*180/np.pi),
                    'Magnification transverse' : self.system.mag,
                    'Magnification axial' : self.system.axial_mag,
                    'MTF base frequency' : self.system.base_freq,
                    'MTF size [pixels]' : self.system.OTF_res,
                    'Optical efficiency' : self.system.tti,
                    'Voxel size [microns]' : self.system.camera.vox*1e6}

        res_data = {'X_res [nm]' : self.system.XYZ_res[0],
                    'Y_res [nm]' : self.system.XYZ_res[1],
                    'Z_res [nm]' : self.system.XYZ_res[2],
                    'X_FWHM [nm]' : self.system.FWHM[0],
                    'Y_FWHM [nm]' : self.system.FWHM[1],
                    'Z_FWHM [nm]' : self.system.FWHM[2]}

        with open(saveloc+'/data.json', 'w') as output:
            json.dump(metadata|res_data, output, indent=4)

        self.TraceSystem.setText('Done!')

    def helpLens(self,param):
        msg = QtWidgets.QMessageBox()
        if param == 'NA':
            msg.setWindowTitle('Numerical Aperture')
            msg.setText('NA of lens')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('To find NA of tube lens, use formula: \n NA_1*f_1=NA_2*f_2')

        elif param == 'RI':
            msg.setWindowTitle('Refractive index')
            msg.setText('RI of immersion medium')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('The immersion medium will be on the side of the lens where the field is focused.')

        elif param == 'rot':
            msg.setWindowTitle('Lens rotation')
            msg.setText('Rotation of the lens around the x-axis')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('If the rotated lens is a focusing lens, all previous components will be rotatet with the lens. \n\nIf there is a refractive index change between two lenses, the material surface will be orthogonal to the optical axis of the rotatet lens.')

        elif param == 'pos':
            msg.setWindowTitle('Lens position')
            msg.setText('Determines the position of the lens in the system')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText("To make the lens appear in the end of the system, use the default value '0'. \nFor another position, give the integer value where the lens should be inputed.")

        x = msg.exec_()

    def helpCamera(self,param):
        msg = QtWidgets.QMessageBox()
        if param == 'pixel':
            msg.setWindowTitle('Pixel count')
            msg.setText('Number of pixels on the camera')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('The number of pixels of the camera in both axes. \nFor highest simulation efficiency, choose a pixel count of 2^n.')
        elif param == 'vox':
            msg.setWindowTitle('Voxel size')
            msg.setText('Voxel size of camera chip')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('Sampling rate of the electric field in the image space. \nThe field of view will be the voxel size times the number of pixels. \nUndersampling might lead to bugs.')
        elif param == 'RMS':
            msg.setWindowTitle('Camera noise')
            msg.setText('RMS of the camera readout')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('Camera noise is modeled as gaussian noise. \nThe standard deviation of the Gaussian distribution is given by the readout RMS (root mean squared) of the camera.')
        elif param == 'offset':
            msg.setWindowTitle('Camera offset')
            msg.setText('Base pixel intensity of the camera readout.')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        elif param == 'OTF':
            msg.setWindowTitle('OTF size')
            msg.setText('Number of pixels in the OTF')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('Size of the OTF array. \nFor highest efficiency, choose a size equal to 2^n.')

        x = msg.exec_()

    def helpLightSheet(self,param):
        msg = QtWidgets.QMessageBox()
        if param == 'opening':
            msg.setWindowTitle('Opening angle')
            msg.setText('Opening angle used to make light-sheet in degrees')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('The opening angle will determine the thickness and length of the light-sheet.')
        elif param == 'lam':
            msg.setWindowTitle('Excitation wavelength')
            msg.setText('Excitation wavelength in nanometer')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('The size of the light-sheet is dependent on the wavelength.')
        elif param == 'pol':
            msg.setWindowTitle('Polarization')
            msg.setText('Light-sheet polarization')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('s-polarization is in the plane of the light-sheet, p-polarization is orthogonal to the light-sheet, and u-polarization is a super position of p- and s-polarization.')

        x = msg.exec_()

    def helpTracing(self,param):
        msg = QtWidgets.QMessageBox()
        if param == 'ensamble':
            msg.setWindowTitle('Orientations')
            msg.setText('Number of orientations in the dipole ensamble')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('The orientations of the dipole is calculated using a Fibonacci lattice.')
        elif param == 'lam':
            msg.setWindowTitle('Emission wavelength')
            msg.setText('Emission wavelength in nanometer')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('The size of the PSF is dependent on the wavelength.')
        elif param == 'SNR':
            msg.setWindowTitle('SNR')
            msg.setText('Signal to noise ratio')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('When given, random Poisson noise will be added to the PSF. \nFor a noise-less sample, set SNR to 0.')
        elif param == 'ani':
            msg.setWindowTitle('Anisotropy')
            msg.setText('Check for anisotropic sample')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDetailedText('If checked, the sample will be made of stationary dipoles that retain some of the excitation polarization in emission.')

        x = msg.exec_()

    def critError(self,spec):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle('Error')
        msg.setText('Invalid '+spec)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        x = msg.exec_()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
