# DOGERAY2
DOGERAY2 is an unbaised somewhat-interactive GPU path tracer written in CUDA C++ that opens GLTF 2.0 files.

![icon](https://user-images.githubusercontent.com/72355251/162350170-a91185f6-28b2-4b1b-a4b1-8ba6ef63fced.png)


## Gallery
<img src="https://user-images.githubusercontent.com/72355251/161661800-df3f2fca-2034-46a6-b702-e8c6b82af441.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161661814-e9ddb8ae-53cd-4c35-a5b0-3c3e4ce50bd9.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161661828-ebbf0937-c3d9-45fe-9c1b-04a4def6ea04.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161661875-63df6d28-b52e-4fe7-b0e1-f6ca7c754d90.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662100-c9f32d2a-4482-48de-8c8b-896572c94c89.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662110-f6a3a737-bb88-47e9-852c-e7659fb993a6.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662123-0e2f2a36-07ac-458d-a53f-ac24bf7c0262.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662151-00874970-4682-4cf4-b90b-a47bcc91224e.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662170-362b9774-f766-42c8-81b0-30444fd884d6.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/162549141-8ffedc0c-7ed8-4272-a1c3-00df69a64f41.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/162549143-1e89f04b-2784-46c4-9563-fba4e2438b30.jpg" width="30%"></img> <img src="https://user-images.githubusercontent.com/72355251/162549147-83afc6f5-67e3-4336-bb46-95769e406b97.jpg" width="30%"></img> 

Find the full size gallery in the [wiki](https://github.com/PhilipPragerUrbina/DOGERAY2/wiki)

## Features
Capable of 4k rendering a few samples per second. Has been tested on scenes with over 15 million triangles.

Some of the current features:
* Responsive GUI with many controls and live render window
* Fully path traced fast GPU accelerated rendering
* JPG image saving
* Window resizing and image scaling
* Interactve and BVH preview
* BVH acceleration strucutre Stackless BVH traversal
* Depth of field 
* PBR and emmisive matirials with Textures
* Smooth shading
* Backround images
* Opens GLTF files with cameras and transformations
* Multithreaded UI




## Quick usage guide:
1. To open a file, just right click on a GLTF or GLB file and open with DOGERAY2.
2. To add a background just put an image file named "background" in the same directory where you are opening the file from.
3. If your model is not rendering correctly, make sure the scale of the model is not overly huge(>1000 units wide).
Check out the Wiki for more info on file formats, common issues, usage, and decisions.

## Dependecies:
1. CUDA
2. SDL2(Included in release)

(Also requires fairly recent Nvidia GPU to run)


## Some learnings from this project:
* Polymorphism and Serialization
* Power of Compiler Optimization
* Using specifications to figure out how to utilize industry standar files
* Casting types and working with raw image data
* Organizing Object oreinted c++
* Using GUI toolkits
* Managing program lifecycle and memory
* Matrix and Vector transformations
* Common code writing practices
* Markdown and Licencing on Github
* Extreme debugging

## Licences:
Full licences can also be found in libraries folder.
* Tinygltf: Copyright (c) 2017 Syoyo Fujita, Aurélien Chatelain and many contributors
  * json.hpp : Copyright (c) 2013-2017 Niels Lohmann. MIT license.
  * base64 : Copyright (C) 2004-2008 René Nyffenegger
* Imgui: Copyright (c) 2014-2022 Omar Cornut
### Other Libraries:
* CUDA
* STB_image and STB_image write
* SDL2
* Linalg

Links to the libraries can be found in the [wiki page](https://github.com/PhilipPragerUrbina/DOGERAY2/wiki/Libraries).


  

## References
The rendering algorithm is based of of Ray Tracing in One Weekend whcih is amazing for learning about how ray tracing works. The stackless bvh traversal algorithm at the heart of the engine is based off of Hachisuka's amazing work.
 * [Ray Tracing in One Weekend Series by Peter Shirley](https://raytracing.github.io/)
 * [Implementing a practical rendering system using GLSL by Toshiya Hachisuka](https://cs.uwaterloo.ca/~thachisu/tdf2015.pdf) 
 ## Why the name?
 DOGERAY2 is a completely new piece of software from the original Dogeray that I wrote a very long time ago. It only keeps the name since I could not come up with a better one.
