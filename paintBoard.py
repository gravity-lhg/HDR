# Author: Haoguang Liu
# Date: 2022.04.12 16:34 PM
# E-mail: liu.gravity@gmail.com

'''
PaintBoard file
'''
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, QColor, QSize
from PyQt5.QtCore import Qt

class PaintBoard(QWidget):

    def __init__(self, Parent=None):
        ''' Set up artboards and related actions '''
        super().__init__(Parent)

        self.__InitData() 
        self.__InitView()
        
    def __InitData(self):
        
        self.__size = QSize(280, 280)
        
        #  creat QPixmap for artboard, size is __size
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.white) # white for background
        
        self.__lastPos = QPoint(0,0) # record last poision of mouse
        self.__currentPos = QPoint(0,0) # record current poision of mouse
        
        self.__painter = QPainter() # create paint tool
        
        self.__thickness = 30  # Brush thickness is 30px
        self.__penColor = QColor("black")   # black for brush
     
    def __InitView(self):
        # Set the size of the interface to __size
        self.setFixedSize(self.__size)
        
    def Clear(self):
        # clear artboard
        self.__board.fill(Qt.white)
        self.update()
    
    def GetContentAsQImage(self):
        # get image from artboard（return QImage type）
        image = self.__board.toImage()
        return image
        
    def paintEvent(self, paintEvent):
        # paint event
        # An instance of QPainter must be used when drawing, here is __painter
        # Drawing is performed between the begin() function and the end() function
        # The parameter of begin(param) should specify the drawing device, that is, where to put the drawing
        # drawPixmap is used to draw objects of type QPixmap
        self.__painter.begin(self)
        # 0,0 is the coordinate of the starting point of the upper left corner of the drawing
        # __board is the drawing to be drawn
        self.__painter.drawPixmap(0,0,self.__board)
        self.__painter.end()
        
    def mousePressEvent(self, mouseEvent):
        # When the mouse is pressed, get the current position of the mouse and save it as the last position
        self.__currentPos =  mouseEvent.pos()
        self.__lastPos = self.__currentPos
        
        
    def mouseMoveEvent(self, mouseEvent):
        # When the mouse moves, update the current position and draw a line between the previous position and the current position
        self.__currentPos =  mouseEvent.pos()
        self.__painter.begin(self.__board)
        
        # set the pencolor and thickness of brush 
        self.__painter.setPen(QPen(self.__penColor,self.__thickness))
            
        # draw line  
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos
                
        self.update() # update the view
        
