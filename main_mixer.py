import image_mixer
from PyQt5 import QtWidgets , QtCore, QtGui
import matplotlib.image as mpimg
import sys
from numpy.fft import fft2, ifft2 , fftshift
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage
import matplotlib.pyplot as plt
from PIL import Image, ImageQt
import logging

# Create and configure logger
logging.basicConfig(level=logging.DEBUG, filename="app.log", format='%(lineno)s - %(levelname)s - %(message)s', filemode='w')

logger = logging.getLogger()

# Fourrier
class Image():
    def __init__(self , img_array):
        self.img_array = img_array   
        self.img_shape =  self.img_array.shape
        self.ft_img = fft2(self.img_array)
        self.shifted_fft = fftshift(self.ft_img)
        
        self.mag = np.abs(self.ft_img)
        self.mag_comp = 20 * np.log(np.abs(self.shifted_fft))  
              
        self.phase = np.angle(self.ft_img)
        self.phase_comp = np.angle(self.shifted_fft)
        
        self.real = np.real(self.ft_img)
        self.real_comp = np.real(self.shifted_fft)
        self.real_comp[self.real_comp <= 0] = 10 ** -100 
        self.real_comp = 20 *np.log(self.real_comp)

        self.imag = np.imag(self.ft_img)
        self.imag_comp = np.imag(self.shifted_fft)
        self.imag_comp[self.imag_comp <= 0] = 10 ** -100
        self.imag_comp = 20 *np.log(self.imag_comp)

        self.unimag = np.ones(self.mag.shape)
        self.uniphase = np.zeros(self.phase.shape)

        self.display_comp = [self.mag_comp ,self.phase_comp ,self.real_comp ,self.imag_comp] 
        self.components =[self.mag ,self.phase ,self.real , self.imag , self.unimag, self.uniphase ]        

        
        

class MainWindow(QtWidgets.QMainWindow , image_mixer.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        logger.info("the application has started")
        self.img = 2 * [0]
        self.combobox =  [self.img1_comp_combobox , self.img2_comp_combobox]
        self.widgets = [self.img1_view , self.img2_view,self.img1_comp, self.img2_comp ,self.output1 , self.output2]
        self.ratio = [0,0]
        self.sliders = [self.comp1_slider, self.comp2_slider]  
        self.combobox_output = [self.output_combobox, self.output_img1_combobox, self.img1_comp_output_combobox, self.output_img2_combobox, self.img2_comp_output_combobox]
        self.img2_comp_output_combobox.setCurrentIndex(1)
        
        for i in range(2):  
            self.mixing_sliders(i)  # connecting mixing sliders
        
        for i in range(5):
            self.mixing_combobox(i) # connecting mixing comboboxes
        
        for i in range(2):
            self.combobox_action(i) # connecting component comboboxes
            
        self.resetButton.clicked.connect(self.reset)
        self.actionOpen.triggered.connect(self.open_image)
        self.widget_configuration()
        self.default()

    # connect component comboboxes to display image
    def combobox_action(self , i):
        self.combobox[i].activated.connect(lambda:self.comp_img(i))
            
    def widget_configuration(self):
        for widget in self.widgets:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def open_image(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File',"", "Image Files (*.png *jpeg *.jpg)")
        
        # Grayscale
        self.img_array = plt.imread(self.file_path)
        R, G, B = self.img_array[:,:,0], self.img_array[:,:,1], self.img_array[:,:,2] 
        self.img_array = (0.2989 * R + 0.5870 * G + 0.1140 * B).T
        
        if self.img[0] == 0 :
            i = 0
            self.combobox[0].setEnabled(True)
            self.resetButton.setEnabled(True)
            self.img[0] = Image(self.img_array)
        elif self.img[1] == 0:
            i = 1
            self.img[1] = Image(self.img_array)
            if ( (np.size(self.img[0].img_array) !=  np.size(self.img[1].img_array))) :
                self.size_pop_up()
                self.img[i] = 0
                self.img_array = 0
                logger.warning("The image size is not suitable")
            else:
                self.en_dis_able(True)
                self.mixing()
                logger.info("tools are enabled")
        
          
            self.comp_img(i) 
            self.display(self.img_array , self.widgets[i] , self.img[0].img_shape)

        
    # Display image in widget    
    def display(self , data , widget , img_shape):
        widget.setImage(data)
        widget.view.setLimits(xMin=0, xMax=img_shape[0], yMin= 0 , yMax= img_shape[1])
        widget.view.setRange(xRange=[0, img_shape[0]], yRange=[0, img_shape[1]], padding=0)
        
    # Enable or Disable UI itmes
    def en_dis_able(self , x):
        self.img2_comp_combobox.setEnabled(x)
        self.output_combobox.setEnabled(x)
        self.output_img1_combobox.setEnabled(x)
        self.img1_comp_output_combobox.setEnabled(x)
        self.output_img2_combobox.setEnabled(x)
        self.img2_comp_output_combobox.setEnabled(x)
        self.comp1_slider.setEnabled(x)
        self.comp2_slider.setEnabled(x)
        self.resetButton.setEnabled(x)
        
    # resetting the app to the launching state
    def reset(self):
        for i in range(6):
            self.widgets[i].clear()
            
        self.en_dis_able(False)
        
        for i in range(2):
            self.combobox[i].setEnabled(False)
            self.img[i] = 0
            
        self.widget_configuration()
        self.default()
        logger.info("the application has been reseted")

    # displaying component on image widget
    def comp_img(self , i):
        if self.img[i] != 0:
            self.display(self.img[i].display_comp[self.combobox[i].currentIndex()] , self.widgets[i + 2] , self.img[0].img_shape)


    # pop-up message
    def size_pop_up(self):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText('Warning!')
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setInformativeText('Image sizes must be the same, please upload another image')
        x = msg.exec_()
        
    def mismatch_pop_up(self):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText('Warning!')
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setInformativeText('Please change the second component')
        x = msg.exec_()

        
       
    
    # connecting the Sliders
    def mixing_sliders(self, i):
        self.sliders[i].valueChanged.connect(lambda: self.mixing())
    def mixing_combobox(self , i):
        self.combobox_output[i].activated.connect(lambda: self.mixing())

    
    def mixing(self):
        
        # disable mismatched items
        if self.img1_comp_output_combobox.currentText() == 'Mag' :
            self.available_items([False,True,False,False,False,True])
        elif self.img1_comp_output_combobox.currentText() == 'Phase' :
            self.available_items([True,False,False,False,True,False])
        elif self.img1_comp_output_combobox.currentText() == 'Real' :
            self.available_items([False,False,False,True,False,False])
        elif self.img1_comp_output_combobox.currentText() == 'Imag' :
            self.available_items([False,False,True,False,False,False])
        elif self.img1_comp_output_combobox.currentText() == 'uniMag' :
            self.available_items([False,True,False,False,False,True])
        elif self.img1_comp_output_combobox.currentText() == 'uniPhase' :
            self.available_items([True,False,False,False,True,False])

        self.image1 = self.img[self.output_img1_combobox.currentIndex()] # the first image to take component from
        self.image2 = self.img[self.output_img2_combobox.currentIndex()] # the second image to take component from

        self.slider1_percent.setText(str(self.sliders[0].value()) + "%") # first slider %
        self.slider2_percent.setText(str(self.sliders[1].value()) + "%") # second slider %
        
        for i in range(2):
            self.ratio[i] = (self.sliders[i].value()/100)
   
        # setting conditions to prevent the app from crashing
        condition1 = (self.img1_comp_output_combobox.currentText() == self.img2_comp_output_combobox.currentText())
        condition2 = (self.img1_comp_output_combobox.currentText() in ['Phase' , 'uniPhase'] and self.img2_comp_output_combobox.currentText() in ['Phase' , 'uniPhase']) or (self.img1_comp_output_combobox.currentText() in ['Mag' , 'uniMag'] and self.img2_comp_output_combobox.currentText() in ['Mag' , 'uniMag'])
        condition3 = (self.img1_comp_output_combobox.currentText() in ['Real' ,'Imag' ] and self.img2_comp_output_combobox.currentText() in ['Phase' ,'Mag','uniPhase','uniMag' ])
        condition4 = (self.img2_comp_output_combobox.currentText() in ['Real' ,'Imag' ] and self.img1_comp_output_combobox.currentText() in ['Phase' ,'Mag','uniPhase','uniMag' ])
        
        if  condition1 or condition3 or condition2 or condition4:
            pass
            self.mismatch_pop_up()
        
        else:
            # component 1 = 70% mag of 1st image + 30% mag of 2nd image
            comp1 = np.add(self.image1.components[self.img1_comp_output_combobox.currentIndex()] * self.ratio[0] , self.image2.components[self.img1_comp_output_combobox.currentIndex()] * (1 - self.ratio[0]))
            logger.info(f"comp1 is {self.img1_comp_output_combobox.currentText()}")
            
            # component 2 = 50% phase of 2nd image + 50% phase of 1st image
            comp2 = np.add(self.image2.components[self.img2_comp_output_combobox.currentIndex()] * self.ratio[1] , self.image1.components[self.img2_comp_output_combobox.currentIndex()] * (1 - self.ratio[1]))
            logger.info(f"comp1 is {self.img2_comp_output_combobox.currentText()}")

            # Constructing complex number (Real + Imaginary)
            if self.img1_comp_output_combobox.currentText() == "Real":
                complex_number = np.add(comp1, comp2 * 1j)
                logger.info("components are added")
                
            # Constructing complex number (Imaginary + Real)    
            elif self.img1_comp_output_combobox.currentText() == "Imag":
                complex_number = np.add(comp1 * 1j, comp2)
                logger.info("components are added")
                
            # Constructing complex number (Magnitude + Phase)
            elif self.img1_comp_output_combobox.currentText() in ["Mag" ,"uniMag"] :
                complex_number = np.multiply(comp1, np.exp(1j * comp2))
                logger.info("components are multiplied")

            # Constructing complex number (Phase + Magnitude)
            elif self.img1_comp_output_combobox.currentText() in ["Phase" ,"uniPhase"] :
                complex_number = np.multiply(comp2, np.exp(1j * comp1))
                logger.info("components are multiplied")
                
                        
            try:
                output = ifft2(complex_number) # Final data to display after mixing
                self.display(output , self.widgets[4 + self.output_combobox.currentIndex()] , self.img[0].img_shape)
            except:
                return None

    # Enable only matched 2nd combobox components with 1st combobox chosen component
    def available_items(self, boolian_list):
        for i in range(len(boolian_list)):
            self.img2_comp_output_combobox.model().item(i).setEnabled(boolian_list[i])
            
    # displaying default image
    def default(self):
        default_image = plt.imread("default/default-image.jpg")
        R, G, B = default_image[:,:,0], default_image[:,:,1], default_image[:,:,2]
        default_image = (0.2989 * R + 0.5870 * G + 0.1140 * B).T
        for i in range(6):
            self.display(default_image , self.widgets[i] , default_image.shape)




def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()