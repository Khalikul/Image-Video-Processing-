import cv2
import numpy as np
from tabulate import tabulate
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compare_images(images):
    num_images = len(images)
    comparisons = [

    ]

    # Iterate through all pairs of images provided by the user
    for i in range(num_images):
        for j in range(i + 1, num_images):
            img1 = images[i]
            img2 = images[j]

            # Convert images to grayscale for SSIM comparison
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # print(img1_gray)
            # print('==========')
            # print(img2_gray)

            # Calculate PSNR and SSIM
            psnr_value = psnr(img1, img2)
            ssim_value = ssim(img1_gray, img2_gray, full=True)[0]

            # Append comparison results
            comparisons.append({
                "Images": f"Image {i+1} vs Image {j+1}",
                "PSNR": psnr_value,
                "SSIM": ssim_value
            })

    return comparisons
#main_code
img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')
img3 = cv2.imread('4.png')
img4 = cv2.imread('5.png')
img5 = cv2.imread('6.png')



# img6 = cv2.imread('_5_4.png')

images = [img1,img3, img2, img4,img5]
results = compare_images(images)

# Print the comparison results in a table
table_data = []
for result in results:
    table_data.append([result["Images"], f"{result['PSNR']:.4f}", f"{result['SSIM']:.6f}"])

print(tabulate(table_data, headers=["Images", "PSNR", "SSIM"], tablefmt="fancy_grid"))
