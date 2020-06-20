import numpy as np
from scipy.spatial.distance import pdist, squareform
from PIL import Image
import matplotlib.pyplot as plt
import os, re

gamma = 1/(231*195)

train_path = os.path.join('Yale_Face_Database', 'Training')
test_path = os.path.join('Yale_Face_Database', 'Testing')
result_path = os.path.join('Result', 'LDA')

# read the input images
def read_images(path):
    dataset, label = list(), list()
    for data in os.listdir(path):
        with Image.open(os.path.join(path, data)).resize((60, 60), Image.ANTIALIAS) as image:
            # with Image.open(os.path.join(path, data)) as image:
            dataset.append(np.array(image).flatten())
            label.append(int(re.findall("\d+", data)[0]))

    # returns a np array with shape (num_imgs)X(num_pixels_per_image)
    return np.array(dataset), np.array(label)

train_images, train_label = read_images(train_path)
train_images = train_images[np.argsort(train_label), :]
train_label = np.sort(train_label)
test_images, test_label = read_images(test_path)
test_images = test_images[np.argsort(test_label), :]
test_label = np.sort(test_label)

def LDA(images, no_dims=25):
    print('Computing Meam...')
    all_mean = np.mean(images, axis=0).reshape(-1, 1).T
    class_mean = []
    for idx in range(15):
        rows = images[9 * idx:9 * (idx + 1), :]
        class_mean.append(np.mean(rows, axis=0))
    class_mean = np.array(class_mean).astype(np.float32)

    print('Computing SW...')
    (n, d) = images.shape
    temp = images.copy().astype(np.float32)
    for data in range(n):
        temp[data, :] -= class_mean[data // 9, :]
    within_class = np.dot(temp.T, temp)

    print('Computing SB...')
    (n, d) = class_mean.shape
    temp = class_mean.copy()
    for data in range(n):
        temp[data, :] -= all_mean[0, :]
    between_class = np.dot(temp.T, temp)
    between_class *= 9

    print('Computing Eigenvector...')
    eigen_values, eigen_vectors = np.linalg.eigh(np.dot(np.linalg.pinv(within_class), between_class))
    print('Computing Principle Component...')
    principle_component = eigen_vectors[:, np.argsort(eigen_values)[::-1][0: no_dims]].real.astype(np.float32)

    return principle_component

def plot_reconstruct(reconstruct, origin, index):
    for idx in index:
        plt.clf()
        plt.imshow(reconstruct[idx].reshape((60, 60)), plt.cm.gray)
        plt.savefig(f"{result_path}/reconstruct_{idx}.png")

        plt.clf()
        plt.imshow(origin[idx].reshape((60, 60)), plt.cm.gray)
        plt.savefig(f"{result_path}/origin_{idx}.png")

def plot_eigenfaces(principle_component):
    for idx in range(len(principle_component)):
        plt.clf()
        plt.imshow(principle_component[idx].reshape((60, 60)), plt.cm.gray)
        plt.savefig(f"{result_path}/eigenfaces_{idx}.png")

def predict(train_data, test_data, k, label):
    prd_result = []
    for data in test_data:
        distance = np.linalg.norm(train_data - data, axis=1)
        idx = np.argsort(distance)
        neighbors = label[idx][:k]
        #neighbors = np.argpartition(distance, -n_neighbors)[-n_neighbors:]# // 9 + 1
        # n_neighbors = distance.argsort()[-n_neighbors:][::-1]
        #print (neighbors)
        prd_result.append(np.bincount(neighbors).argmax())

    return np.array(prd_result)

def make_kernel(images, kernel):
    if kernel == 'rbf':
        similarity = squareform(pdist(images), 'sqeuclidean')
        gram = np.exp(-gamma * similarity)
    elif kernel == 'linear':
        gram = np.dot(images, images.T)
    return gram

def kernel_LDA(train, test, no_dims=25):
    kernels = ['linear', 'rbf']
    for kernel in kernels:
        print('Computing Meam...')
        all_mean = np.mean(train, axis=0).reshape(-1, 1).T
        class_mean = []
        for idx in range(15):
            rows = train[9 * idx:9 * (idx + 1), :]
            class_mean.append(np.mean(rows, axis=0))
        class_mean = np.array(class_mean).astype(np.float32)

        print('Computing SW...')
        (n, d) = train.shape
        temp = train.copy().astype(np.float32)
        for data in range(n):
            temp[data, :] -= class_mean[data // 9, :]
        within_class = make_kernel(temp.T, kernel)

        print('Computing SB...')
        (n, d) = class_mean.shape
        temp = class_mean.copy()
        for data in range(n):
            temp[data, :] -= all_mean[0, :]
        between_class = make_kernel(temp.T, kernel)
        between_class *= 9

        print('Computing Eigenvector...')
        eigen_values, eigen_vectors = np.linalg.eigh(np.dot(np.linalg.pinv(within_class), between_class))
        print('Computing Principle Component...')
        principle_component = eigen_vectors[:, np.argsort(eigen_values)[::-1][0: no_dims]].real.astype(np.float32)

        train_low_images = np.dot(train, principle_component)
        test_low_images = np.dot(test, principle_component)

        predict_result = predict(train_low_images, train_low_images, 5, train_label)
        print(
            f'Predict result of Training data set using {kernel} kernel is {len(predict_result[predict_result == train_label]) / len(train_label)}')
        predict_result = predict(train_low_images, test_low_images, 5, train_label)
        print(
            f'Predict result of Testing data set using {kernel} kernel is {len(predict_result[predict_result == test_label]) / len(test_label)}')

if __name__ == "__main__":
    principle_component = LDA(train_images)

    low_images = np.dot(train_images, principle_component)
    high_images = np.dot(low_images, principle_component.T)

    index = np.random.randint(len(train_images), size=10)
    plot_reconstruct(high_images, train_images, index)

    plot_eigenfaces(principle_component.T)

    train_low_images = np.dot(train_images, principle_component)
    test_low_images = np.dot(test_images, principle_component)
    predict_result = predict(train_low_images, train_low_images, 5, train_label)
    # predict_result = predict(train_low_images, train_low_images, 1, train_label)
    print(f'Predict result of Training data set is {len(predict_result[predict_result == train_label]) / len(train_label)}')
    predict_result = predict(train_low_images, test_low_images, 5, train_label)
    # predict_result = predict(train_low_images, test_low_images, 1, train_label)
    print(f'Predict result of Testing data set is {len(predict_result[predict_result == test_label]) / len(test_label)}')

    kernel_LDA(train_images, test_images)