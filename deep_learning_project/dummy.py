
import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

if __name__ == '__main__':
    test_dir = './test_images_bootstrap'

    save_dir = './someimages/'

    transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0,), std=(1,))])

    test_data = torchvision.datasets.ImageFolder(
        test_dir, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=True, num_workers=1)

    for idx, data in enumerate(test_loader):
        image, label = data
        image_array = image.cpu().numpy()
        image_array = np.array(image_array*255, dtype='int8')
        image_to_save = Image.fromarray(image_array.reshape(36, 36))
        image_to_save.show()
        image_to_save.save(save_dir+"img"+str(idx)+".pgm")
        # break
        # indice de la valeur max (0 pas face, 1, c'est face)
        #image.save(save_dir, "img"+idx+".pgm")
