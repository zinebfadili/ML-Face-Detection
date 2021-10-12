from torchsampler.imbalanced import *

threshold = 0.8
train_dir = './train_images_bootstrap'
test_dir = './test_images_bootstrap'

while threshold > 0:
    print(threshold)

    # get equal amount of data
    # train
    # test on the textures
    # keep the images that have over threshold in 1
    # inject those in train in non faces
    # repeat

    # get equal amount of data
    # load traindata
    train_data = torchvision.datasets.ImageFolder(
        train_dir, transform=transform)
    # get indices of train data and shuffle them
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)

    # create balancer sampler for train data
    train_sampler = ImbalancedDatasetSampler(train_data, indices=indices_train)

    # load train and test data
    train_data = torchvision.datasets.ImageFolder(
        train_dir, transform=transform)
    # these are our own textures
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    
    

    threshold -= 0.2
