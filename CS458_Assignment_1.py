#
# file  CS490_Assignment_1.py
# brief Purdue University Fall 2022 CS490 robotics Assignment 1 -
#       Gaussian Discriminant Analysis
# date  2022-09-01
#

# you can only import modules listed in the handout
import sys
import time

import numpy as np
import roipoly as roi
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_label_and_image(path, region):
    for filename in os.listdir(region):
        # print(filename)
        root_path, extension_path = os.path.splitext(filename)
        full_path = os.path.join(path, root_path + '.png')
        img = plt.imread(full_path)
        plt.imshow(img)
        mask = np.load(os.path.join(region, root_path + '.npy'), allow_pickle=True)
        plt.imshow(mask, alpha=0.3)
        plt.show()


def store_segmented_img(path, image_buf):
    mpimg.imsave(path, image_buf)


# hand label region related functions
# **************************************************************************************************
# hand label pos/neg region for training data
# write regions to files, this function should not return anything
def label_training_dataset(training_path, region_path):
    if not os.path.exists(region_path):
        os.makedirs(region_path)
    for filename in os.listdir(training_path):
        # print(filename)
        root_path, extension_path = os.path.splitext(filename)
        if os.path.exists(os.path.join(region_path, root_path + '.npy')):
            continue
        full_path = os.path.join(training_path, filename)
        img = plt.imread(full_path)
        plt.imshow(img)
        roi_shape = roi.RoiPoly(color='r')
        mask = roi_shape.get_mask(img[:, :, 0])
        np.save(os.path.join(region_path, root_path), mask)


# hand label pos region for testing data
# write regions to files, this function should not return anything
def label_testing_dataset(training_path, region_path):
    if not os.path.exists(region_path):
        os.makedirs(region_path)
    for filename in os.listdir(training_path):
        # print(filename)
        root_path, extension_path = os.path.splitext(filename)
        if os.path.exists(os.path.join(region_path, root_path + '.npy')):
            continue
        full_path = os.path.join(training_path, filename)
        img = plt.imread(full_path)
        plt.imshow(img)
        roi_shape = roi.RoiPoly(color='r')
        mask = roi_shape.get_mask(img[:, :, 0])
        np.save(os.path.join(region_path, root_path), mask)
    pass


# **************************************************************************************************


# import labeled regions related functions
# **************************************************************************************************


def load_img_and_mask(img_buf, mask):
    img_height = img_buf.shape[0]
    img_width = img_buf.shape[1]
    features = np.zeros((img_height, img_width, img_buf.shape[2]))
    labels = np.zeros((img_height, img_width))
    for h in range(img_height):
        for w in range(img_width):
            pixel_rgb = img_buf[h, w, :]
            features[h, w] = pixel_rgb
            labels[h, w] = mask[h, w]
    return features, labels



# import pre hand labeled region for trainning data
def import_pre_labeled_training(training_path, region_path):
    features_set, labels_set = [], []
    for filename in os.listdir(training_path):
        root_path, extension_path = os.path.splitext(filename)
        full_path = os.path.join(training_path, filename)
        img = plt.imread(full_path)
        mask = np.load(os.path.join(region_path, root_path + '.npy'))
        features, labels = load_img_and_mask(img, mask)
        features_set.append(features)
        labels_set.append(labels)

    return features_set, labels_set



# import per hand labeled region for testing data
def import_pre_labeled_testing(testing_path, region_path):
    features_set, labels_set = [], []
    for filename in os.listdir(testing_path):
        root_path, extension_path = os.path.splitext(filename)
        full_path = os.path.join(testing_path, filename)
        img = plt.imread(full_path)
        mask = np.load(os.path.join(region_path, root_path + '.npy'))
        features, labels = load_img_and_mask(img, mask)
        features_set.append(features)
        labels_set.append(labels)

    return features_set, labels_set


# **************************************************************************************************


# main GDA training functions
# **************************************************************************************************
# features: ndarray pixels
# labels: mask
# 1 covariance matrix for both class, but different mean and prior
def train_GDA_common_variance(features, labels):
    features_array = np.array(features)
    features_flattened = features_array.reshape(-1, 3)
    labels_array = np.array(labels)
    labels_flattened = labels_array.reshape(-1, 1)

    total_len = len(features_flattened)
    pos_len, neg_len = 0, 0
    pos_pixel, neg_pixel = np.zeros(3), np.zeros(3)
    for (i, pixel) in enumerate(features_flattened):
        if labels_flattened[i]:
            pos_pixel += pixel
            pos_len += 1
        else:
            neg_pixel += pixel
            neg_len += 1

    prior = [pos_len / total_len, neg_len / total_len]
    mu = [pos_pixel / pos_len, neg_pixel / neg_len]  # [pos_mean, neg_mean]
    # print(f"mu: {mu}")

    covariance_mat = np.zeros((3, 3))
    for (i, sample) in enumerate(features_flattened):
        diff = sample - mu[not labels_flattened[i].item()]  # flip 0 and 1
        covariance_mat += np.outer(diff, diff)

    cov = covariance_mat / total_len

    # print(prior, mu, cov)
    return prior, mu, cov


def train_GDA_variable_variance(features, labels):
    features_array = np.array(features)
    features_flattened = features_array.reshape(-1, 3)
    labels_array = np.array(labels)
    labels_flattened = labels_array.reshape(-1, 1)


    total_len = len(features_flattened)
    pos_len, neg_len = 0, 0
    pos_pixel, neg_pixel = np.zeros(3), np.zeros(3)
    for (i, pixel) in enumerate(features_flattened):
        if labels_flattened[i]:
            pos_pixel += pixel
            pos_len += 1
        else:
            neg_pixel += pixel
            neg_len += 1

    prior = [pos_len / total_len, neg_len / total_len]
    mu = [pos_pixel / pos_len, neg_pixel / neg_len]  # [pos_mean, neg_mean]

    covariance_mat = [np.zeros((3, 3)), np.zeros((3, 3))]
    for (i, sample) in enumerate(features_flattened):
        label_idx = not labels_flattened[i].item()  # flip 0 and 1
        diff = sample - mu[label_idx]
        covariance_mat[label_idx] += np.outer(diff, diff)

    cov = (covariance_mat[0] / pos_len, covariance_mat[1] / neg_len)  # pos_cov, neg_cov

    return prior, mu, cov


# **************************************************************************************************

def linear_discriminant_analysis(x, pos_prior, neg_prior, pos_mu, neg_mu, common_cov):
    inv_cov = np.linalg.inv(common_cov)

    term1 = (pos_mu - neg_mu).T @ inv_cov @ x
    term2 = -0.5 * (pos_mu.T @ inv_cov @ pos_mu - neg_mu.T @ inv_cov @ neg_mu)

    ln_prior_ratio = np.log(pos_prior / neg_prior)

    return (term1 + term2 + ln_prior_ratio).item()


def quadratic_discriminant_analysis(x, pos_prior, neg_prior, pos_mu, neg_mu, pos_cov, neg_cov):
    inv_cov_pos = np.linalg.inv(pos_cov)
    inv_cov_neg = np.linalg.inv(neg_cov)

    term1 = x.T @ (inv_cov_neg - inv_cov_pos) @ x
    term2 = 2 * ((inv_cov_pos @ pos_mu) - (inv_cov_neg @ neg_mu)).T @ x
    term3 = neg_mu.T @ inv_cov_neg @ neg_mu - pos_mu.T @ inv_cov_pos @ pos_mu

    ln_cov_ratio = np.log(np.linalg.det(neg_cov) / np.linalg.det(pos_cov))
    ln_prior_ratio = np.log(pos_prior / neg_prior)

    return (term1 + term2 + term3 + ln_cov_ratio + 2 * ln_prior_ratio).item()


# GDA testing and accuracy analyis functions
# **************************************************************************************************
# assign labels using trained GDA parameters for testing features
def predict(testing_features, theta, mu, cov) :
    (pos_prior, neg_prior) = theta
    (pos_mu, neg_mu) = mu
    masks = []
    algo_type = True

    if not isinstance(cov, tuple): # is qda
        for i, feature in enumerate(testing_features):
            img_height = feature.shape[0]
            img_width = feature.shape[1]
            mask = np.zeros((img_height, img_width), dtype=bool)
            masked_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            for h in range(img_height):
                for w in range(img_width):
                    pixel_rgb = feature[h, w, :]
                    cls = linear_discriminant_analysis(pixel_rgb, pos_prior, neg_prior, pos_mu, neg_mu, cov)
                    mask[h, w] = cls > 0
                    masked_img[h, w] = [255, 0, 0] if cls > 0 else [0, 0, 0]
            masks.append(mask)
        algo_type = True
    else:
        (pos_cov, neg_cov) = cov
        for (i, feature) in enumerate(testing_features):
            img_height = feature.shape[0]
            img_width = feature.shape[1]
            mask = np.zeros((img_height, img_width), dtype=bool)
            masked_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            for h in range(img_height):
                for w in range(img_width):
                    pixel_rgb = feature[h, w, :]
                    cls = quadratic_discriminant_analysis(pixel_rgb, pos_prior, neg_prior, pos_mu, neg_mu, pos_cov, neg_cov)
                    mask[h, w] = cls > 0
                    masked_img[h, w] = [255, 0, 0] if cls > 0 else [0, 0, 0]
            masks.append(mask)
        algo_type = False

    return (algo_type, masks)

# print precision/call for both classes to console
#
# example console printout:
# GDA with common variance:
# precision of label 0: xx.xx%
# recall of label 0:    xx.xx%
# precision of label 1: xx.xx%
# recall of label 1:    xx.xx%
# GDA with variable variance:
# precision of label 0: xx.xx%
# recall of label 0:    xx.xx%
# precision of label 1: xx.xx%
# recall of label 1:    xx.xx%
#
# list of masks
def accuracy_analysis(predicted_labels, ground_truth_labels):
    # precision: ratio of true positives to all positive predictions made
    # recall: ratio of true positives to all actual positive cases
    algo_type, labels = predicted_labels
    total_true_positives, total_false_positives = 0, 0
    total_false_negatives, total_true_negatives = 0, 0

    for i, sample in enumerate(labels):
        height = sample.shape[0]
        width = sample.shape[1]
        gt_label = ground_truth_labels[i]

        for h in range(height):
            for w in range(width):
                gt_pixel = gt_label[h, w]
                predicted_pixel = sample[h, w]

                if predicted_pixel and gt_pixel:
                    total_true_positives += 1
                elif predicted_pixel and not gt_pixel:
                    total_false_positives += 1
                elif not predicted_pixel and gt_pixel:
                    total_false_negatives += 1
                elif not predicted_pixel and not gt_pixel:
                    total_true_negatives += 1

    pos_precision = total_true_positives / (total_true_positives + total_false_positives)
    pos_recall = total_true_positives / (total_true_positives + total_false_negatives)
    neg_precision = total_true_negatives / (total_true_negatives + total_false_negatives)
    neg_recall = total_true_negatives / (total_true_negatives + total_false_positives)

    if algo_type:
        print("GDA with common variance:")
    else:
        print("GDA with variable variance:")
    # label 0 is positive class
    # label 1 is negative class
    print(f"precision of label 0: {pos_precision*100:.2f}%")
    print(f"recall of label 0:    {pos_recall*100:.2f}%")
    print(f"precision of label 1: {neg_precision*100:.2f}%")
    print(f"recall of label 1:    {neg_recall*100:.2f}%")





# **************************************************************************************************


if __name__ == '__main__':
    #Please read this block before coding
    #**********************************************************************************************
    #caution: when you submit this file, make sure the main function is unchanged otherwise your
    #         grade will be affected because the grading script is designed based on the current
    #         main function
    #
    #         Also, do not print unnecessary values other than the accuracy analysis in the console

    #Labeling during runtime can be very time-consuming during the debugging phase.
    #Also, it is hard to ensure the labelings are consistent during each testing run.
    #Thus, we do this in separate stages.
    #First, implement all the functions and uncomment the three lines in the data loader block.
    #Then, revert the main function back to what it is used to be and start implementing the rest
    #**********************************************************************************************


    #data loader used to generate your labeling
    #ideally this block should only be called once
    #**********************************************************************************************
    #label_training_dataset('trainset', 'train_region')
    #label_testing_dataset('testset', 'test_region')
    #sys.exit(1)
    #**********************************************************************************************


    #import your generated labels from saved data
    #**********************************************************************************************
    training_features, training_labels = import_pre_labeled_training('trainset', 'train_region')
    testing_features, ground_truth_labels = import_pre_labeled_testing('testset', 'test_region')
    #**********************************************************************************************


    #GDA with common varianve
    #**********************************************************************************************
    prior, mu, cov = train_GDA_common_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************


    #GDA with variable variance
    #**********************************************************************************************
    prior, mu, cov = train_GDA_variable_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************
