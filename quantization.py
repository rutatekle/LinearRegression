import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import io
import imageio


def quantize(raster, k):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_

    quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1]))
    # file_name=("/Users/ayohannes/Desktop/Quantized/quanitized_image.png")
    # io.imsave(file_name,quantized_raster)
    return quantized_raster

def show_orginal_image(filename):
    raster = imageio.imread(filename)
    print("\nOrginal Image")
    io.imshow(raster)
    io.show()
    return raster

def show_quantized_image(k,raster):
    print("\nQuantized Image: " + str(k))
    quantized_raster=quantize(raster,k)
    plt.imshow(quantized_raster / 255.0)
    plt.draw()
    plt.show()


raster1=show_orginal_image('https://utd-class.s3.amazonaws.com/clustering/image2.jpg')
raster2=show_orginal_image('https://utd-class.s3.amazonaws.com/clustering/image3.jpg')
raster3=show_orginal_image('https://utd-class.s3.amazonaws.com/clustering/image4.jpg')

show_quantized_image(5,raster1)
show_quantized_image(5,raster2)
show_quantized_image(20,raster3)



