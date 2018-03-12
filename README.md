# Perspective Invariant Feature Tranform (PIFT)
## Description
  PIFT is a local RGB-D feature for RGB-D images, which is invaiant to different views. The depth information was used to project the feature patch to its spatial tangent plane to make it consistent under different views. It also helps to filter out the fake keypoints which are unstable in 3D space. The binary descriptors are generated in the feature patches using the color coding method based on the color information.
  
  This project cantians the basic implementation and the interfaces of PIFT, in the files of pift.h and pift.cpp. The demonstration of extracting PIFT features and feature maching is also provided in this project.
  
  Maintainer: NuBot workshop, NUDT China - http://nubot.trustie.net and https://github.com/nubot-nudt
## Requirements
  OpenCV 2.X
  
  PCL 1.7.X
## Make and Run
  cmake .
  
  make
  
  ./pift_demo
