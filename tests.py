from counter import HairCounter
from PIL import Image
import numpy


def process_img():
    img = numpy.array(Image.open('./images/2.png'))
    
    hair_counter = HairCounter(img)
    hair_counter.run()

    print({'message': 'success', 'strand_number': hair_counter.hair_number, 'used_area': hair_counter.used_area, 'base64_image': hair_counter.getBase64Image()})

if __name__ == "__main__":
    process_img()