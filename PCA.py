import numpy as np
from scipy.spatial.distance import pdist, squareform
from PIL import Image
import matplotlib.pyplot as plt
import os, re

gamma = 1/(231*195)

train_path = os.path.join('Yale_Face_Database', 'Training')
test_path = os.path.join('Yale_Face_Database', 'Testing')
result_path = os.path.join('Result', 'PCA')

# read the input images
def read_images(path):
    dataset, label = list(), list()
    for data in os.listdir(path):
        with Image.open(os.path.join(path, data)) as image:
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

def PCA(images, no_dims=25):
    (n, d) = images.shape
    images = images - np.tile(np.mean(images, 0), (n, 1))
    eigen_values, eigen_vectors = np.linalg.eigh(np.dot(images, images.T))
    print("Computing Eigenvalues...")
    principle_component = eigen_vectors[:, np.argsort(eigen_values)[::-1][0: no_dims]].real.astype(np.float32)
    principle_component = np.dot(images.T, principle_component).real.astype(np.float32)
    norm = np.linalg.norm(principle_component, axis=0)
    principle_component = np.divide(principle_component, norm)

    return principle_component

def plot_reconstruct(reconstruct, origin, index):
    for idx in index:
        plt.clf()
        plt.imshow(reconstruct[idx].reshape((231, 195)), plt.cm.gray)
        plt.savefig(f"{result_path}/reconstruct_{idx}.png")

        plt.clf()
        plt.imshow(origin[idx].reshape((231, 195)), plt.cm.gray)
        plt.savefig(f"{result_path}/origin_{idx}.png")

def plot_eigenfaces(principle_component):
    for idx in range(len(principle_component)):
        plt.clf()
        plt.imshow(principle_component[idx].reshape((231, 195)), plt.cm.gray)
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

def kernel_PCA(images, no_dims=25):
    kernels = ['linear', 'rbf']
    for kernel in kernels:
        K = make_kernel(images, kernel)
        N = K.shape[0]
        one = np.ones((N, N)) / N
        Kc = K - np.dot(one, K) - np.dot(K, one) + np.dot(np.dot(one, K), one)
        eigen_values, eigen_vectors = np.linalg.eigh(Kc)
        principle_component = eigen_vectors[:, np.argsort(eigen_values)[::-1][0: no_dims]].real.astype(np.float32)

        low_images = np.dot(K, principle_component)

        train_low_images = low_images[:len(train_images), ]
        test_low_images = low_images[len(train_images):, ]

        predict_result = predict(train_low_images, train_low_images, 5, train_label)
        print(
            f'Predict result of Training data set using {kernel} kernel is {len(predict_result[predict_result == train_label]) / len(train_label)}')
        predict_result = predict(train_low_images, test_low_images, 5, train_label)
        print(
            f'Predict result of Testing data set using {kernel} kernel is {len(predict_result[predict_result == test_label]) / len(test_label)}')

if __name__ == "__main__":
    principle_component = PCA(train_images)
    mean = np.mean(train_images, 0)
    high_images = np.dot(train_images - mean, np.dot(principle_component, principle_component.T)) + mean

    index = np.random.randint(len(train_images), size=10)
    plot_reconstruct(high_images, train_images, index)

    plot_eigenfaces(principle_component.T)

    train_low_images = np.dot(train_images - mean, principle_component)
    test_low_images = np.dot(test_images - mean, principle_component)
    predict_result = predict(train_low_images, train_low_images, 5, train_label)
    print(f'Predict result of Training data set is {len(predict_result[predict_result == train_label]) / len(train_label)}')
    predict_result = predict(train_low_images, test_low_images, 5, train_label)
    print(f'Predict result of Testing data set is {len(predict_result[predict_result == test_label]) / len(test_label)}')

    images = np.concatenate((train_images, test_images), axis=0)
    kernel_PCA(images)