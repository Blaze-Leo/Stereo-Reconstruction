# Depth-Reconstruction

Divides the image into textures on assumption that same coloured parts that are adjacent together are flat object faces. Then caluculates the distance from the original pixel to the other stereo image and displays creates a disparity map that can be used to create a 3d model. Pretty good accuracy, but can be made far better if not bound the run time cost. Optimization could be done, but won't change the runtime drastically.
