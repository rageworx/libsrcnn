# libsrcnn
### Super-Resolution imaging with Convolutional Neural Network
A stand-alone library for Super-Resolution, Non-OpenCV model related in these projects:
* https://github.com/rageworx/SRCNN_OpenCV_GCC
* https://github.com/shuwang127/SRCNN_Cpp.

## Introduction
This is an open source project from original of this:
**SRCNN_Cpp** is a C++ Implementation of Image Super-Resolution using SRCNN which is proposed by Chao Dong in 2014.
 - If you want to find the details of SRCNN algorithm, please read the paper:  

   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, in Proceedings of European Conference on Computer Vision (ECCV), 2014
 - If you want to download the training code(caffe) or test code(Matlab) for SRCNN, please open your browse and visit http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html for more details.
 - And thank you very much for Chao's work in SRCNN.

## Additional references
* Fast resizing methods for BiCubic filtering 
    * [Free Image Project](http://freeimage.sourceforge.net/)
	* [librawprocessor](https://github.com/rageworx/librawprocessor)
    * [fl_imgtk](https://github.com/rageworx/fl_imgtk)

 
## Features
* No OpenCV.
* Compile availed for almost of POSIX.
* Simply optimized for basic OpenMP.
* Works well even withou OpenMP ( as like MacOSX )
* Not support M$VC.

## Supporting platforms
* Windows 32, 64 (MinGW-W64)
* Linux
* Embedded Linux, ARM, AARCH64
* MacOSX ( clang, llvm )

## Latest Changes
### Verison 0.1.5.18
* Supports variable filters for interpolation.
    1. Nearest
    1. Bilinear
    1. Bicubic
    1. Lanczos-3
    1. B-Spline

## Previous Changes
### Version 0.1.4.17
* Bug fixed for color space conversion.

## License
* Follows original source GPLv2, but this project is LGPLv3.

## Requirements
* Your G++.

## How to build ?
* build with ```make -f Makefile.{your platform}```.
