# CV-Hand-gesture-control
Tinkering around with basic convolutions with OpenCV and Haar Cascade classifiers to detect hand gestures and face through webcam.
Depending on the hand gesture, a different filter is applied to the webcam video capture. Try a frontal facing fist or hand palm gesture.

# Requirements:
You need to download OpenCV packages for python. You can run **pip install opencv-contrib-python** to do so.

# To do:
- Detect when user touches his/her face, keeping a record of number of times face is touched and image snap shots of when face is touched. Can help develop good hygienic habits.
- This requires a better model for detecting the hand

# Future:
- Train your own Haar Cascade

# Sources:
Credit to the following repositories for pretrained Haar Cascade classifiers:
https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV/tree/master/data/haarcascades
https://github.com/Aravindlivewire/Opencv/blob/master/haarcascade/aGest.xml
