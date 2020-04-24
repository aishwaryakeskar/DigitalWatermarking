from watermarking import watermarking
from DE import get_de_values
import test


image_name = "baboon"
jpg_image_type = ".jpg"
image_type = ".jpg"
watermark_image_name = "watermark1" + image_type
watermarked_image = "watermarked_" + image_name + image_type


values = get_de_values()

Watermarking = watermarking(level=3, x=values[2])

# Gaussian noise attack

Watermarking.watermark()
test.add_gaussian_noise(watermarked_image)
test.plot_image(image_name + image_type, 'gaussian_noise_attack' + jpg_image_type, 'Gaussian Attack')
test.print_data("Gaussian Noise Attack", "gaussian_noise_attack" + jpg_image_type, image_name + image_type)

Watermarking.extracted(image_path="gaussian_noise_attack" + jpg_image_type,
                       extracted_watermark_path="watermark_extracted_gaussian" + jpg_image_type)
test.calculate_psnr_nc(img1=watermark_image_name, img2="watermark_extracted_gaussian" + jpg_image_type)
print("For the Watermark image")
test.print_data("Gaussian Noise Attack", watermark_image_name, "watermark_extracted_gaussian" + jpg_image_type)
test.plot_image(watermark_image_name, 'watermark_extracted_gaussian' + jpg_image_type, 'Gaussian Attack')

# Salt pepper noise

test.add_salt_pepper_noise(watermarked_image, prob=0.01)
test.plot_image(watermarked_image, "salt_pepper_noise_attack" + jpg_image_type, 'Salt Pepper Attack')
test.print_data("Salt and Pepper Noise Attack", "salt_pepper_noise_attack" + jpg_image_type, watermarked_image)

Watermarking.extracted(image_path="salt_pepper_noise_attack" + jpg_image_type,
                       extracted_watermark_path="watermark_extracted_salt_pepper" + jpg_image_type)
test.calculate_psnr_nc(img1=watermark_image_name, img2="watermark_extracted_salt_pepper" + jpg_image_type)
print("For the Watermark image")
test.print_data("Salt and Pepper Attack", "watermark_extracted" + jpg_image_type, "watermark_extracted_salt_pepper" + jpg_image_type)
test.plot_image(watermark_image_name, "watermark_extracted_salt_pepper" + jpg_image_type, "Salt and Pepper Attack")

# Rotation attack

test.add_rotation_test(0, watermarked_image)
test.plot_image(watermarked_image, "rotation_attack" + jpg_image_type, 'Rotation Attack')
test.print_data("Rotation Attack", "rotation_attack" + jpg_image_type, watermarked_image)

Watermarking.extracted(image_path="rotation_attack" + jpg_image_type, extracted_watermark_path="watermark_extracted_rotation" + jpg_image_type)
test.calculate_psnr_nc(img1=watermark_image_name, img2="watermark_extracted_rotation" + jpg_image_type)
print("For the Watermark image")
test.print_data("Rotation Attack", "watermark_extracted" + jpg_image_type, "watermark_extracted_rotation" + jpg_image_type)
test.plot_image(watermark_image_name, "watermark_extracted_rotation" + jpg_image_type, "Rotation Attack ")

test.add_rotation_test(20, watermarked_image)
test.plot_image(watermarked_image, "rotation_attack" + jpg_image_type, 'Rotation Attack')
test.print_data("Rotation Attack", "rotation_attack" + jpg_image_type, watermarked_image)

Watermarking.extracted(image_path="rotation_attack" + jpg_image_type, extracted_watermark_path="watermark_extracted_rotation" + jpg_image_type)
test.calculate_psnr_nc(img1=watermark_image_name, img2="watermark_extracted_rotation" + jpg_image_type)
print("For the Watermark image")
test.print_data("Rotation Attack", "watermark_extracted" + jpg_image_type, "watermark_extracted_rotation" + jpg_image_type)
test.plot_image(watermark_image_name, "watermark_extracted_rotation" + jpg_image_type, "Rotation Attack ")

# Cropping attack

test.rescale_test(watermarked_image)
test.plot_image(watermarked_image, "cropped_attack" + jpg_image_type, 'Cropping Attack')
test.print_data("Crop Attack", "cropped_attack" + jpg_image_type, watermarked_image)

Watermarking.extracted(image_path="cropped_attack" + jpg_image_type, extracted_watermark_path="watermark_extracted_crop" + jpg_image_type)
test.calculate_psnr_nc(img1=watermark_image_name, img2="watermark_extracted_crop" + jpg_image_type)
print("For the Watermark image")
test.print_data("Cropping Attack", "watermark_extracted" + jpg_image_type, "watermark_extracted_crop" + jpg_image_type)
test.plot_image(watermark_image_name, "watermark_extracted_crop" + jpg_image_type, "Cropping Attack ")

# Poisson noise attack

test.add_poisson_noise(watermarked_image)
test.plot_image(watermarked_image, "poisson_noise_attack" + jpg_image_type, 'Poisson Noise Attack')
test.print_data("Poisson Noise Attack", "poisson_noise_attack" + jpg_image_type, watermarked_image)

Watermarking.extracted(image_path="poisson_noise_attack" + jpg_image_type,
                       extracted_watermark_path="watermark_extracted_poisson_noise" + jpg_image_type)
test.calculate_psnr_nc(img1=watermark_image_name, img2="watermark_extracted_poisson_noise" + jpg_image_type)
print("For the Watermark image")
test.print_data("Poisson Noise Attack", watermark_image_name, "watermark_extracted_poisson_noise" + jpg_image_type)
test.plot_image(watermark_image_name, "watermark_extracted" + jpg_image_type, "Poisson Noise Attack ")

# Compression attack

test.compression_test(watermarked_image)
test.plot_image(watermarked_image, "compressed_watermarked_lena" + jpg_image_type, 'Compression Attack')
test.print_data("Compression Attack", "compressed_watermarked_lena" + jpg_image_type, watermarked_image)

Watermarking.extracted(image_path="Compressed_watermarked_lena" + jpg_image_type,
                       extracted_watermark_path="watermark_extracted_compression_attack" + jpg_image_type)
test.calculate_psnr_nc(img1=watermark_image_name, img2="watermark_extracted_compression_attack" + jpg_image_type)
print("For the Watermark image")
test.print_data("Compression Attack", "watermark_extracted" + jpg_image_type, "watermark_extracted_compression_attack" + jpg_image_type)
test.plot_image(watermark_image_name, "watermark_extracted_compression_attack" + jpg_image_type, "Compression Attack")



test.testing(watermarked_image)

test.rotate(watermarked_image)
