import torch
from net import *
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import cv2


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[0] / scale)
        h = int(image.shape[1] / scale)
        #image = imutils.resize(image, width=w)
        image = cv2.resize(image, (h, w))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def pyramid_sliding_window_detection(net, image, scale, winW, winH, stepSize):
    # Store the initial image before resize, it will be used for the final printing
    faces_img = image.copy()

    # loop over the image pyramid
    # all_detected_faces : contains for each pyramid level the scaling factor and the detected faces corresponding to
    # pyramid level
    all_detected_faces = []
    for resized in pyramid(image, scale=scale):
        detected_faces = []
        curr_scale_factor = image.shape[0] / resized.shape[0]
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # We use the 36*36 window to match the net's img input size
            resized_tensor = torch.from_numpy(window)
            # Transform the 500*500 (2d) img to a 4d tensor (the additional 2 dimensions contain no information)
            # tensor shape is now [1,1,500,500]
            resized_tensor = resized_tensor[None, None, :, :]
            # Feed the network the input tensor
            output = net(resized_tensor)

            # We only register faces with a prob higher than 0.99 to avoid false positives
            # we have already added softmax as a last activation function on our model
            if output[0][1] >= 0.95:
                # print(output[0][1])
                detected_faces.append((x, y))

        # Add the detected faces and the corresponding factors to the all_faces variable
        all_detected_faces.append([curr_scale_factor, detected_faces])

    # We use the non_max_supp algorithm to delete overlaping bounding boxes
    # to avoid detecting the same face multiple times
    for j in range(len(all_detected_faces)):
        # all_detected_faces[j][1]->detected faces of the i-pyramid-level
        for i in range(len(all_detected_faces[j][1])):
            # in this line we both :
            # - change the tuple from a 2d (startX, startY) to a 4d (startX, startY, endX, endY)
            # - multiply each number of the tuple by the current scale factor
            all_detected_faces[j][1][i] = (
                all_detected_faces[j][1][i][0] *
                all_detected_faces[j][0], all_detected_faces[j][1][i][1] *
                all_detected_faces[j][0]
            ) + (
                (all_detected_faces[j][1][i][0] + winW)*all_detected_faces[j][0], (
                    all_detected_faces[j][1][i][1] + winH)*all_detected_faces[j][0]
            )
    # print(all_detected_faces)
    # Concatenate detected faces into the same array
    final_detected_faces = all_detected_faces
    return final_detected_faces


if __name__ == '__main__':
    net = Net()
    net.load_state_dict(torch.load("./model_with_bootstrap.pth"))
    #src_image_path = "./scale_images/247147411_3719854041572098_7124613502422578930_n.pgm"
    src_image_path = "./cropped_nirvana.pgm"
    face_coord = []
    with Image.open(src_image_path) as image:
        width, height = image.size
        source_img = Image.open(src_image_path).convert("RGBA")
        draw = ImageDraw.Draw(source_img)
        for i in np.linspace(1.5, 6, 10):
            scale = height/(36*i)
            print(scale)
            faces = pyramid_sliding_window_detection(
                net, np.array(image, dtype='float32'), scale, 36, 36, 6)
            # print(faces)
            for face in faces[1][1]:
                # print(face)
                draw.rectangle(
                    ((face[0], face[1]), (face[2], face[3])), outline="red")

        rgb_im = source_img.convert('RGB')
        out_file = "./rectangle_image.jpg"
        rgb_im.save(out_file, "JPEG")
