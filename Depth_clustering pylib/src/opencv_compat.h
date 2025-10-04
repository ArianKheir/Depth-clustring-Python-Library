// opencv_compat.h
#ifndef OPENCV_COMPAT_H
#define OPENCV_COMPAT_H

#include <opencv2/opencv.hpp>

// For OpenCV 4.x compatibility
#if CV_MAJOR_VERSION >= 4
    #include <opencv2/imgproc.hpp>
    #include <opencv2/imgcodecs.hpp>
    #include <opencv2/highgui.hpp>
    #include <opencv2/core.hpp>
#else
    #include <opencv/cv.h>
    #include <opencv/highgui.h>
#endif

#endif // OPENCV_COMPAT_H
