import imageio
from PIL import Image
import numpy as np

# Open the input video file with imageio
input_video = imageio.get_reader("./Gifs/animation_104.gif")

# Create an empty list to store the output frames
output_frames = []

# Iterate over the frames of the input video and resize each frame
for i in range(len(input_video)):
    frame = input_video.get_data(i).copy()
    # Resize the frame to a smaller size
    new_size = (frame.shape[1] // 3, frame.shape[0] // 3)
    resized_frame = Image.fromarray(frame).resize(new_size)
    
    # Add the resized frame to the output writer
    output_frames.append(resized_frame)
    
    # Print progress every 10 frames
    if i % 10 == 0:
        print(f"Processed {i} frames")
        # if i > 1:
        #     break

# Save the output frames as a new video file
output_video = imageio.get_writer("output.gif")
for frame in output_frames:
    output_video.append_data(np.asarray(frame, dtype=np.uint8))
output_video.close()
