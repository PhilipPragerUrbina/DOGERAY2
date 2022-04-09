# DOGERAY2
DOGERAY2 is an unbiased somewhat-interactive GPU path tracer written in CUDA C++ that opens GLTF 2.0 files.

![icon](https://user-images.githubusercontent.com/72355251/162350170-a91185f6-28b2-4b1b-a4b1-8ba6ef63fced.png)


## Gallery
<img src="https://user-images.githubusercontent.com/72355251/161661800-df3f2fca-2034-46a6-b702-e8c6b82af441.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/161661814-e9ddb8ae-53cd-4c35-a5b0-3c3e4ce50bd9.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/161661828-ebbf0937-c3d9-45fe-9c1b-04a4def6ea04.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/161661875-63df6d28-b52e-4fe7-b0e1-f6ca7c754d90.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662100-c9f32d2a-4482-48de-8c8b-896572c94c89.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662110-f6a3a737-bb88-47e9-852c-e7659fb993a6.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662123-0e2f2a36-07ac-458d-a53f-ac24bf7c0262.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/161662151-00874970-4682-4cf4-b90b-a47bcc91224e.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162549141-8ffedc0c-7ed8-4272-a1c3-00df69a64f41.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162549143-1e89f04b-2784-46c4-9563-fba4e2438b30.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162549147-83afc6f5-67e3-4336-bb46-95769e406b97.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162583880-4f7e930f-4384-46bc-a9ce-1181a2ae10e8.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162583892-f89354ce-2407-4026-a38d-313d1de2beb0.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162583921-73057d53-c4e1-40bd-a9ad-d870aae5faf6.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162583929-636a2440-d9b3-4a3f-bf91-878943f9e6b2.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162585184-0b043923-725e-4407-87c2-25ddb2e190a7.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162585189-95d1a4f3-e68c-44db-8f46-23a4189839b3.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162585343-412d219a-d6c1-4a86-9816-c9feb368ba5a.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162585376-2f82c0f3-7e7a-41e5-ba9d-bc06960ed174.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/72355251/162585199-734fb710-e7b8-43ab-ae91-8151700219ca.jpg" width="18%"></img> 

Find the full size gallery in the [wiki](https://github.com/PhilipPragerUrbina/DOGERAY2/wiki)

## Features
Capable of 4k rendering a few samples per second. Here is a scene with over 15 million triangles:
![15mil glb](https://user-images.githubusercontent.com/72355251/162584919-96f79e24-76ce-4963-9f6f-f5905a8c823a.jpg)


Some of the current features:
* Responsive GUI with many controls and live render window
* Fully path traced fast GPU accelerated rendering
* Multiple JPG image saving
* Window resizing and image scaling
* Interactve and BVH preview
* BVH acceleration structure Stackless BVH traversal
* Depth of field 
* PBR and emmissive materials with Textures
* Smooth shading
* Background images
* Opens GLTF files with cameras and transformations
* Multithreaded UI

User interface:
 ![gui](https://user-images.githubusercontent.com/72355251/162583963-e8663e4b-5189-4469-9e63-9efa44b36539.JPG)

Interactive render preview:
![preview](https://user-images.githubusercontent.com/72355251/162583979-1e1c9e95-cdbd-494f-9e96-c7bf035a0fee.JPG)

Bounding volume hierarchy view:
![sanfoeerd3 glb](https://user-images.githubusercontent.com/72355251/162584006-dba195ad-08f9-415b-8fa9-f92ea05f610a.jpg)

Glossy and emmisive materials, Depth of field:
![helmet2 glb_](https://user-images.githubusercontent.com/72355251/162585121-f3149d6e-6a7e-4fe8-bc15-10ffa39a7c5f.jpg)
![MetalRoughSpheres glb](https://user-images.githubusercontent.com/72355251/162585490-2f2cfeae-60f5-4ea9-9e7a-2327152a4e19.jpg)

Quick video [demo](https://youtu.be/OJ85TVysA_E)




## Quick usage guide:
1. To open a file, just right click on a GLTF or GLB file and open with DOGERAY2.
2. To add a background just put an image file named "background" in the same directory where you are opening the file from.
3. If your model is not rendering correctly, make sure the scale of the model is not overly huge(>1000 units wide).
Check out the Wiki for more info on file formats, common issues, usage, and decisions.

## Dependencies:
1. CUDA
2. SDL2(Included in release)

(Also requires a fairly recent Nvidia GPU to run)


## Some learnings from this project:
* Polymorphism and Serialization
* Power of compiler Optimization
* Using specifications to figure out how to utilize industry standard files
* Casting types and working with raw image data
* Organizing Object oriented  c++
* Using GUI toolkits
* Managing program lifecycle and memory
* Matrix and Vector transformations
* Common code writing practices
* Markdown and Licencing on Github
* Extreme debugging

## Licenses:
Full licenses can also be found in libraries folder.
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
The rendering algorithm is based on Ray Tracing in One Weekend which is amazing for learning about computer graphics. The stackless BVH traversal algorithm at the heart of the engine is based on Hachisuka's amazing work.
 * [Ray Tracing in One Weekend Series by Peter Shirley](https://raytracing.github.io/)
 * [Implementing a practical rendering system using GLSL by Toshiya Hachisuka](https://cs.uwaterloo.ca/~thachisu/tdf2015.pdf) 
 ## Why the name?
 DOGERAY2 is a completely new piece of software from the original Dogeray that I wrote a very long time ago. It only keeps the name since I could not come up with a better one.
