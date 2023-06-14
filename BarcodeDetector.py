import sys
import cv2
import imageIO.png
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pyzbar import pyzbar
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import *

###################################
##   Code for Barcode Detection  ##
###################################
# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):


    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:

'''Question 1: Convert to Greyscale and Normalise'''
# Computes a greyscale representation from the red, green and blue channels.
# Week 10, Question 4
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(0, image_height): 
        for j in range(0, image_width):
            
            r = 0.299*pixel_array_r[i][j]
            g = 0.587*pixel_array_g[i][j]
            b = 0.114*pixel_array_b[i][j]
            
            greyscale_pixel_array[i][j] = round(r+g+b)
    
    return greyscale_pixel_array

# Computes a contrast stretching from the minimum and maximum values of the input pixel array to the full 8 bit range of values between 0 and 255. 
# Week 10, Question 5
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    
    new_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    tuple = computeMinAndMaxValues(pixel_array, image_width, image_height)
    max = tuple[0]
    min = tuple[1]
    if (max == min):
        return new_pixel_array
    
    for i in range(0, image_height): 
        for j in range(0, image_width):
            new_pixel_array[i][j] = round((pixel_array[i][j] - min) * (255/(max-min)))
    
    return new_pixel_array

# Computes minimum and maximum values. This function is used in the function "scaleTo0And255AndQuantize".
# Week 10, Question 5
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    
    max = 0
    min = 255
    
    for i in range(0, image_height): 
        for j in range(0, image_width):
            pixel_value = pixel_array[i][j]
            if (max < pixel_value):
                max = pixel_value
            elif (min > pixel_value):
                min = pixel_value
                
    return (max, min)

'''Question 2: Image Gradient Method'''
# Computes and returns an image of the vertical edges.
# Week 11, Question 1
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    
    sobel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    
    for r in range(0, image_height):
        for c in range(0, image_width):
            
            value = 0.0
            if(not  ((r == 0) | (r == image_height-1) | (c == 0) | (c == image_width-1))):
                
                l_top = pixel_array[r-1][c-1]
                l_mid = pixel_array[r][c-1]
                l_btm = pixel_array[r+1][c-1]
                
                
                r_top = pixel_array[r-1][c+1]
                r_mid = pixel_array[r][c+1] 
                r_btm = pixel_array[r+1][c+1]
                
                
                
                value = (-1.0)*(l_top) + (-2.0)*(l_mid) + (-1.0)*(l_btm) + (1.0)*(r_top) + (2.0)*(r_mid) + (1.0)*(r_btm)
                value /= 8.0
                value = abs(value)
            
            sobel_array[r][c] = value
    
    return sobel_array

# Computes and returns an image of the vertical edges.
# Week 11, Question 2
def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    
    sobel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    
    for r in range(0, image_height):
        for c in range(0, image_width):
            
            value = 0.0
            if(not  ((r == 0) | (r == image_height-1) | (c == 0) | (c == image_width-1))):
                
                l_top = pixel_array[r-1][c-1]
                l_mid = pixel_array[r][c-1]
                l_btm = pixel_array[r+1][c-1]
                
                m_top = pixel_array[r-1][c]
                m_btm = pixel_array[r+1][c]
                
                r_top = pixel_array[r-1][c+1]
                r_mid = pixel_array[r][c+1] 
                r_btm = pixel_array[r+1][c+1]
                
                
                
                value = (1.0)*(l_top) + (2.0)*(m_top) + (1.0)*(r_top) + (-1.0)*(l_btm) + (-2.0)*(m_btm) + (-1.0)*(r_btm)
                value /= 8.0
                value = abs(value)
            
            sobel_array[r][c] = value
    
    return sobel_array

# Computes the absolute value of the difference between two images.
# Not in coderunner
def computeAbsoluteDifference(pixel_array1, pixel_array2, image_width, image_height):

    pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(0, image_height):
        for c in range(0, image_width):
            pixel_array[r][c] = abs(pixel_array1[r][c] - pixel_array2[r][c])

    return pixel_array

'''Question 3: Gaussian Filter'''
# Computes and returns a Gaussian filtered image of the same size as the input image. 
# Week 11, Question 5
def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    answer_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # One pixel
    if ((image_height == 1) & (image_width == 1)):
        return pixel_array
        
    for r in range(0, image_height):
        for c in range(0, image_width):
            
            left = c-1
            right = c+1
            top = r-1
            btm = r+1
            
            
            # Top left pixel
            if ((r == 0) & (c == 0)):
                left = c
                top = r
                
            # Top right pixel
            elif ((r == 0) & (c == image_width-1)):
                top = r
                right = c
            
            # Bottom left pixel
            elif ((r == image_height-1) & (c == 0)):
                btm = r
                left = c
            
            # Bottom right pixel
            elif ((r == image_height-1) & (c == image_width-1)):
                btm = r
                right = c
            
            # Top row
            elif (r==0):
                top = r
            
            # Bottom row
            elif (r == image_height-1):
                btm = r
                
            # Left column
            elif (c == 0):
                left = c
            
            # Right column
            elif (c == image_width-1):
                right = c
            
            l_top = pixel_array[top][left]
            l_mid = pixel_array[r][left]
            l_btm = pixel_array[btm][left]
            
            m_top = pixel_array[top][c]
            m_mid = pixel_array[r][c]
            m_btm = pixel_array[btm][c]
            
            r_top = pixel_array[top][right]
            r_mid = pixel_array[r][right] 
            r_btm = pixel_array[btm][right]
            
            value = l_top + 2*l_mid + l_btm + 2*m_top + 4*m_mid + 2*m_btm + r_top + 2*r_mid + r_btm
            value /= 16.0
        
            answer_array[r][c] = value
    
    return answer_array

'''Question 4: Threshold the Image'''
# Computes threshold value for an image
# Not in coderunner
def computeThresholdValue(pixel_array, image_width, image_height):
    
    histogram = [0] * 256
    
    for r in range(0, image_height):
        for c in range(0, image_width):
                
                value = int(pixel_array[r][c])
                histogram[value] = histogram[value] + 1

    threshold = 0
    previous_threshold = thresholdCalculator(histogram, 0)

    while (previous_threshold != threshold):
        previous_threshold = threshold
        threshold = thresholdCalculator(histogram, threshold)

    return threshold

## 
def thresholdCalculator(histogram, threshold):

    sum_below_threshold = 0
    px_below_threhold = 0
    sum_above_threshold = 0
    px_above_threhold = 0

    for value in range(0, threshold):
        sum_below_threshold += value*histogram[value]
        px_below_threhold += histogram[value]


    for value in range(threshold, 256):
        sum_above_threshold += value*histogram[value]
        px_above_threhold += histogram[value]


    average_above = 0
    average_below = 0

    if (px_below_threhold != 0):
        average_below = sum_below_threshold/px_below_threhold
    if (px_above_threhold != 0):
        average_above = sum_above_threshold/px_above_threhold

    return round((average_below + average_above)/2)

# Computes the thresholding of an image.
# Not in coderunner
def computeThresholding(pixel_array, image_width, image_height, threshold):
    threshold_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for r in range(0, image_height):
        for c in range(0, image_width):
            
            if (pixel_array[r][c] < threshold):
                threshold_array[r][c] = 0
            else:
                threshold_array[r][c] = 255
    
    return threshold_array

'''Question 5: Erosion and Dilation'''
# Computes and returns the eroded image.
# Week 12, Question 2
def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    
    eroded_array = createInitializedGreyscalePixelArray(image_width, image_height, 1)
    
    for j in range(image_height): 
        for i in range(image_width):
            
            
            if (pixel_array[j][i] == 0):
                
                for y in range(-2, 3): 
                    for x in range(-2, 3):
                        
                        py = j + y
                        px = i + x
                        
                        if (0 <= px < image_width) and (0 <= py < image_height):
                            eroded_array[py][px] = 0
            
            if ((i==0) or (j==0) or (i == image_width-1) or (j == image_height-1)):
                eroded_array[j][i] = 0
            
                
    
    return eroded_array

# Computes and returns the dilated image. 
# Week 12, Question 1
def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    
    dilated_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for j in range(image_height): 
        for i in range(image_width):
            
            if (pixel_array[j][i] > 0):
                
                for y in range(-2, 3): 
                    for x in range(-2, 3):
                        
                        py = j + y
                        px = i + x
                        
                        if (0 <= px < image_width) and (0 <= py < image_height):
                            dilated_array[py][px] =1

    return dilated_array


'''Question 6: Connected Component Analysis'''
# Takes the binary pixel array, image width and image height as an input and performs the single pass queue-based connected component labeling algorithm.
# Week 12, Question 3
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    ccimg = createInitializedGreyscalePixelArray(image_width, image_height, 0)
    ccsizes = {}

    currentLabel = 0
    for j in range(image_height): 
        for i in range(image_width):
            
            # Finds the start of a region
            if (pixel_array[j][i] > 0):
                currentSize = 1
                
                currentLabel += 1
                
                # Create a queue and mark visited by changing to black
                queue = Queue()
                queue.enqueue((j,i))
                
                while(not (queue.isEmpty())):
                    
                    # First element in queue
                    dequeue = queue.dequeue()
                    y = dequeue[0]
                    x = dequeue[1]
                    ccimg[y][x] = currentLabel
                    
                    
                    
                    # Mark as visited
                    pixel_array[y][x] = 0
                    
                    # Queue connected and unvisited pixels
                    if (x+1 < image_width):
                        if (pixel_array[y][x+1] > 0):
                            pixel_array[y][x+1] = 0
                            queue.enqueue((y,x+1))
                            currentSize += 1
                        
                    if (x-1 >= 0):
                        if (pixel_array[y][x-1] > 0):
                            pixel_array[y][x-1] =0
                            queue.enqueue((y,x-1))
                            currentSize += 1
                            
                    if (y+1 < image_height):
                        if (pixel_array[y+1][x] > 0):
                            pixel_array[y+1][x]=0
                            queue.enqueue((y+1,x))
                            currentSize += 1
                            
                    if (y-1 >= 0):
                        if (pixel_array[y-1][x] > 0):
                            pixel_array[y-1][x] =0
                            queue.enqueue((y-1,x))
                            currentSize += 1
                        
                ccsizes.update({currentLabel: currentSize})
                
    tuple = (ccimg, ccsizes)
    return tuple

# Analyse connected components and return key of barcode
def connectedComponentAnalysis(ccimg, ccsizes, image_width, image_height):

    # Sort ccsizes
    ccsizes = {k: v for k, v in sorted(ccsizes.items(), key=lambda item: item[1])}

    # Analyse starting with largest component
    for key in reversed(ccsizes):

        tuple = findMinAndMaxCC(ccimg, key, image_width, image_height)
        minx = tuple[0]
        miny = tuple[1]
        maxx = tuple[2]
        maxy = tuple[3]

        ratio = (maxy-miny+1)/(maxx-minx+1)

        # Ratio between 18/10 and 10/18
        if (10/18 <= ratio) and (ratio <= 18/10):
            
            return key
    
    return 0

# Find min and max
def findMinAndMaxCC(ccimg, key, image_width, image_height):

    minx = -1
    miny = -1
    maxx = -1
    maxy = -1
    
    for r in range(0, image_height):
        for c in range(0, image_width):
            
            # If column is smaller than min x
            if ((ccimg[r][c] == key) and ((c < minx) or (minx == -1))):
                minx = c

            if ((ccimg[r][c] == key) and ((r < miny) or (miny == -1))):
                miny = r

            if ((ccimg[r][c] == key) and ((c > maxx) or (maxx == -1))):
                maxx = c
                
            if ((ccimg[r][c] == key) and ((r > maxy) or (maxy == -1))):
                maxy = r
    
    # Min x, min y, max x, max y
    tuple = (minx, miny, maxx, maxy)
    return tuple

'''Extension'''
def findBarcode(filename): 
    input_filename = filename

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # STUDENT IMPLEMENTATION here

    px_array_greyscale = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    px_array_stretch = scaleTo0And255AndQuantize(px_array_greyscale, image_width, image_height)
    px_array_vertical = computeVerticalEdgesSobelAbsolute(px_array_stretch, image_width, image_height)
    px_array_horizontal = computeHorizontalEdgesSobelAbsolute(px_array_stretch, image_width, image_height)
    px_array_difference = computeAbsoluteDifference(px_array_vertical, px_array_horizontal, image_width, image_height)
    px_array_gaussian = computeGaussianAveraging3x3RepeatBorder(px_array_difference, image_width, image_height)
    px_array_gaussian = computeGaussianAveraging3x3RepeatBorder(px_array_gaussian, image_width, image_height)
    px_array_gaussian = computeGaussianAveraging3x3RepeatBorder(px_array_gaussian, image_width, image_height)
    px_array_gaussian = computeGaussianAveraging3x3RepeatBorder(px_array_gaussian, image_width, image_height)
    px_array_gaussian = computeGaussianAveraging3x3RepeatBorder(px_array_gaussian, image_width, image_height)

    px_array_threshold = computeThresholding(px_array_gaussian, image_width, image_height, threshold= computeThresholdValue(px_array_gaussian, image_width, image_height))
    px_array = computeErosion8Nbh5x5FlatSE(px_array_threshold, image_width, image_height)
    px_array = computeErosion8Nbh5x5FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh5x5FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh5x5FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh5x5FlatSE(px_array, image_width, image_height)
    px_array1 = computeDilation8Nbh5x5FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh5x5FlatSE(px_array1, image_width, image_height)

    tuple = computeConnectedComponentLabeling(px_array, image_width, image_height)
    ccimg = tuple[0]
    ccsizes = tuple[1]
    barcode_key = connectedComponentAnalysis(ccimg, ccsizes, image_width, image_height)
    tuple = findMinAndMaxCC(ccimg, barcode_key, image_width, image_height)
    
    return tuple


###################################
##   Application for Extension   ##
###################################
class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.tabWidget = QTabWidget()
        self.tabWidget.currentChanged.connect(self.onTabChanged)

        realtimeTab = RealtimeTab()
        uploadImageTab = UploadImageTab()


        self.tabWidget.addTab(realtimeTab, 'Realtime Detection')
        self.tabWidget.addTab(uploadImageTab, 'Upload Image')


        vbox = QVBoxLayout()
        vbox.addWidget(self.tabWidget)
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.setWindowTitle('Barcode Detector')
        self.setWindowIcon(QIcon("icon.png"))
        self.move(1200, -800)
        self.resize(800, 600)
        self.show()

    def onTabChanged(self):
        currentTab = self.tabWidget.currentWidget()
        currentTab.refreshWindowOnLoad()

    def disableTrainTab(self, shouldDisable):
        self.tabWidget.setTabEnabled(1, shouldDisable)

from abc import ABC, abstractmethod

class TabBaseAbstractClass():

    @abstractmethod
    def refreshWindowOnLoad(self):
        pass

class UploadImageTab(QWidget):
    def __init__(self):
        super().__init__()
        self.filename = "images/Barcode1.png"

        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        uploadImageButton = QPushButton("Upload image")
        vbox.addWidget(uploadImageButton)
        uploadImageButton.clicked.connect(self.upload_image)


        # Plot of barcode detection
        # Create a figure and axes for plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(2, 2)

        vbox.addWidget(self.canvas)
        self.setLayout(vbox)
    
    def upload_image(self):
        # Open file dialog to choose the image file
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
        filename, _ = file_dialog.getOpenFileName(self, "Upload Image", "", "Image Files (*.png *.jpg *.jpeg)")
        
        if filename:
            print("Uploaded image:", filename)
            self.filename = filename
            self.update_plot()

    def update_plot(self):

        (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(self.filename)

        # Clear the previous plot
        for ax in self.axes.flat:
            ax.clear()

        # Plot the input red channel of the image
        self.axes[0, 0].set_title('Input red channel of image')
        self.axes[0, 0].imshow(px_array_r, cmap='gray')

        # Plot the input green channel of the image
        self.axes[0, 1].set_title('Input green channel of image')
        self.axes[0, 1].imshow(px_array_g, cmap='gray')

        # Plot the input blue channel of the image
        self.axes[1, 0].set_title('Input blue channel of image')
        self.axes[1, 0].imshow(px_array_b, cmap='gray')

        # Draw a bounding box as a rectangle into the input image
        image = mpimg.imread(self.filename)
        self.axes[1, 1].set_title('Final image of detection')
        self.axes[1, 1].imshow(image, cmap='gray')

        tuple = findBarcode(self.filename)
        minx = tuple[0]
        miny = tuple[1]
        maxx = tuple[2]
        maxy = tuple[3]
        rect = Rectangle((minx, miny), maxx - minx, maxy - miny, linewidth=1, edgecolor='g', facecolor='none')
        self.axes[1, 1].add_patch(rect)

        # Update the canvas
        self.canvas.draw()

    def refreshWindowOnLoad(self):
        pass

class RealtimeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        gridLayout = QGridLayout()
        self.setLayout(gridLayout)

        # Continue Button
        captureButton = QPushButton("Capture")
        captureButton.clicked.connect(self.capture_image)
        captureButton.setStyleSheet("font-size: 16px")
        gridLayout.addWidget(captureButton, 1, 0)

        # QLabel to display the camera feed
        self.camera_label = QLabel()
        gridLayout.addWidget(self.camera_label, 0, 0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    def refreshWindowOnLoad(self):
        pass

    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), aspectRatioMode=True))

            # Perform barcode detection
            barcodes = pyzbar.decode(frame)
            for barcode in barcodes:
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                barcode_data = barcode.data.decode("utf-8")
                barcode_type = barcode.type
                text = "{} ({})".format(barcode_data, barcode_type)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert the frame to RGB format for displaying in QLabel
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create QImage from the frame data
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)

    def capture_image(self):
        # Pause the camera feed
        self.timer.stop()

        # Capture photo
        ret, frame = self.camera.read()

        # Open file dialog to choose the save location
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png)")

        if file_path:
            # Retrieve the current frame from the camera feed
            if ret:
                # Save the frame as an image file
                cv2.imwrite(file_path, frame)
                QMessageBox.information(self, "Image Saved", "The image has been saved successfully.")
        
        # Resume the camera feed
        self.timer.start()


    def closeEvent(self, event):
        self.camera.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())