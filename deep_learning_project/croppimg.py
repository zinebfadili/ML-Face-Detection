from PIL import Image, ImageFont, ImageDraw, ImageEnhance

if __name__ == '__main__':
    src_image_path = "./scale_images/247147411_3719854041572098_7124613502422578930_n.pgm"
    with Image.open(src_image_path) as img:
        print(img.size)
        width, height = img.size
        cropped = img.crop((0, 0, width, 900))  # (left, upper, right, lower)
        cropped.convert('L').save(
            "texture_pgm_36_36.pgm")
        print(cropped.size)
