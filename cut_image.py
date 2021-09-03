import os
import glob
import argparse
from PIL import Image

class SplitImage():

    def split_image(self,input_dir, output_dir, w, h):    # set w and h as the required resolution
        Image.MAX_IMAGE_PIXELS = None # to avoid image size warning

        imgdir = input_dir

        filelist = [f for f in glob.glob(imgdir + "**/*.jpg", recursive=True)]
        savedir = output_dir

        start_pos = start_x, start_y = (0, 0)
        cropped_image_size = w, h 

        for file in filelist:
            img = Image.open(file)
            width, height = img.size

            frame_num = 1
            for col_i in range(0, width, w):
                for row_i in range(0, height, h):
                    crop = img.crop((col_i, row_i, col_i + w, row_i + h))
                    name = os.path.basename(file)
                    name = os.path.splitext(name)[0]
                    save_to= os.path.join(savedir, name+"_{:03}.jpg")
                    crop.save(save_to.format(frame_num))
                    frame_num += 1

def main():
    parser = argparse.ArgumentParser(description='Script to cut images into specified size', add_help=False)
    parser.add_argument('-i', '--input_dir', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-w','--width', type=int, default=None)
    parser.add_argument('-h', '--height', type=int, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    width = args.width
    height = args.height

    cut = SplitImage()
    cut.split_image(input_dir, output_dir, width, height)

if __name__ == '__main__':
    main()
