import imageio
from PIL import Image
import os

def concatenate_gifs_horizontally(gif_paths, output_path):
    # Read all GIFs
    gifs = [imageio.mimread(gif_path) for gif_path in gif_paths]

    # Ensure all GIFs have the same number of frames
    num_frames = min(len(gif) for gif in gifs)
    gifs = [gif[:num_frames] for gif in gifs]

    # Concatenate frames horizontally
    concatenated_frames = []
    for frame_idx in range(num_frames):
        frames = [Image.fromarray(gif[frame_idx]) for gif in gifs]
        widths, heights = zip(*(frame.size for frame in frames))
        total_width = sum(widths)
        max_height = max(heights)

        new_frame = Image.new('RGBA', (total_width, max_height))
        x_offset = 0
        for frame in frames:
            new_frame.paste(frame, (x_offset, 0))
            x_offset += frame.width

        concatenated_frames.append(new_frame)

    # Save the concatenated GIF
    concatenated_frames[0].save(output_path, save_all=True, append_images=concatenated_frames[1:], loop=0)


# Example usage
exp_dir = '/data/a/bowenz/MAVIN/assets/exp3'
gifs = ['mavin', 'dynami', 'dynami-vid', 'seine', 'seine-vid']
gif_paths = [os.path.join(exp_dir, f'{gif}.gif') for gif in gifs]
output_path = 'concatenated.gif'
concatenate_gifs_horizontally(gif_paths, output_path)
