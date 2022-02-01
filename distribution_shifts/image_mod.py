
#OLD, do not use this anymore

# import torch 
# import torchvision
# from PIL import Image, ImageFilter
# import numpy as np 
# import random 

# def insert_random_noise(im, stddev=100):
#     # noise = random.normal(loc=0.0, scale=sigma, size=(250, 250))
#     w, h = im.size
#     for i in range(w):
#         # print(i, "/", w)
#         for j in range(h): 
#             add_noise_one_pixel(im, i, j, stddev)

#     return im 
    
# def add_noise(x, stddev):
#     return min(max(0, int(random.normalvariate(x,stddev))), 255)

# def add_noise_one_pixel(im, i, j, stddev):
#     x, y, z = im.getpixel((i,j))
#     im.putpixel((i, j), (add_noise(x, stddev), add_noise(y, stddev), add_noise(z, stddev)))
# def gaussian_blur(im, rad): 
#     return im.filter(ImageFilter.GaussianBlur(radius = rad))

# def main(): 
#     path = "/home/thlarsen/lander_climbing.jpeg"
#     with Image.open(path) as im:
#         # im.show()
#         im = gaussian_blur(im, 10)
#         # insert_random_noise(im, 100)
#         im.rotate(270).save('/home/thlarsen/blur.jpeg')

# if __name__ == "__main__":
#     main()


