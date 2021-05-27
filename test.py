# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# from numpy.fft import fft2 , ifft2
# import numpy as np
# img_array = mpimg.imread("index.jpeg") 
# ft =fft2(img_array)
# mag = np.abs(ft)
# # phase = np.exp(1j * np.angle(ft))
# real = np.real(ft)
# imag = np.imag(ft) * 1j
# ahmed = np.add(real , imag)
# img = np.real(ifft2(imag))
# print(img.min())


# plt.imshow(img)
# plt.show()
import numpy as np
from matplotlib import pyplot as plt

img1 = plt.imread('index.jpeg',0)



image1_fourir  = np.fft.fft2(img1)
image1_fourir_fshift = np.fft.fftshift(image1_fourir)
image1_magnitude_spectrum = 20 * np.log(np.abs(image1_fourir))
image1_magnitude_spectrum_shifted = 20 * np.log(np.abs(image1_fourir_fshift))
image1_phase_spectrum = np.angle(image1_fourir)
image1_real_part = np.real(image1_fourir)
image1_img_part = np.imag(image1_fourir)
plt.subplot(231),plt.imshow(img1, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(image1_magnitude_spectrum_shifted, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(image1_phase_spectrum, cmap = 'gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()