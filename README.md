# Barcode Detector

### Introduction
As an extension to barcode detection (in compsci 373), I created an application using PyQt5 for barcode detection and recognition using the computer’s camera in real-time or through image upload. The purpose of this application is so users can easily see the computer detect barcodes in both real-time and through static images. Without this application, users would to take a picture on a separate application, save it, and then edit the filename in the main assignment to see if the barcode can be detected. This app has two main parts (“real-time detection/camera” and “upload image”), separated by tabs at the top.
### Real-time Detection & Camera
When the application is run, the real-time detection tab is displayed. It has a live camera created using the openCV library, which accesses the camera feed and continuously captures frames. The detection of barcodes was implemented using the Pyzbar library, which reads and decodes the barcodes in real-time. The camera can also capture and save images as png to files which can then be used in the “Upload Image” tab.
### Upload Image
Initially, the plots are empty on this tab until the user uploads images from files for barcode detection. Once an image is selected, the image is processed, and the barcode is detected using the pipeline from the main part of this assignment and plots them.
### Improvements
In the future, this application's user interface could be enhanced by adding more colour and styling. Making the application more visually appealing would allow the users to interact with the application more efficiently, but more importantly, increasing the speed of the barcode detection pipeline would improve the application.
