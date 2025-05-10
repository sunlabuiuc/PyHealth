# Using Kaggle Dataset from https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
# Original DICOM format dataest is 150GB, this Kaggle dataset is in JPEG and is about 6GB
# More accessible to work with, much less data size but still same amount of patient samples
# Built to work with the Kaggle dataset, modified https://github.com/lotterlab/task_word_explainability
# Put Kaggle files in ./data inside the project

from scipy.spatial import distance
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
import pydicom
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
import os
import imageio
import cv2


def process_dcm(file_path):
    ds = pydicom.dcmread(file_path)
    im = ds.pixel_array
    im = im.astype(float)
    # simple normalization, convert to RGB
    im = im / im.max()
    im2 = np.zeros(list(im.shape) + [3])
    for i in range(3):
        im2[:, :, i] = im
    im = (255 * im2).astype(np.uint8)

    return im


def create_clip_feature_mat(file_list, clip_model, preprocess_fxn):
    X = np.zeros((len(file_list), 512)) # 512 is feature dimension
    for i, f in tqdm(enumerate(file_list), total=len(file_list)):
        if '.dcm' in f:
            im = Image.fromarray(process_dcm(f))
        else:
            im = Image.open(f)
        im = preprocess_fxn(im).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(im)
        X[i] = image_features[0].cpu()

    return X

def fit_words(train_df, test_df, device, word_list, save_dir, save_tag):
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    X_train = create_clip_feature_mat(train_df.file_path.values, clip_model, preprocess_fxn)

    # # Truncate samples to match lengths
    min_length = min(len(X_train), len(train_df))
    X_train = X_train[:min_length]
    train_df = train_df.iloc[:min_length]

    # Fit the classifier
    if train_df.label.isnull().any():
        print("Warning: NaN values found in train_df.label. Removing rows with NaN values.")
        # Drop rows with NaN values in train_df.label
        train_df = train_df.dropna(subset=['label']).reset_index(drop=True)
        # Truncate X_train to match the new length of train_df
        X_train = X_train[:len(train_df)]
    classifier = LogisticRegression(random_state=0, C=1, max_iter=1000, verbose=1, fit_intercept=False)
    classifier.fit(X_train, train_df.label.values)

    # Create X_test before scoring
    X_test = create_clip_feature_mat(test_df.file_path.values, clip_model, preprocess_fxn)

    # Ensure there are no NaN values in the label column
    if test_df['label'].isnull().any():
        print("Warning: NaN values found in test_df['label']. Handling them...")
        test_df = test_df.dropna(subset=['label'])  # Option 1: Drop rows with NaN values
        # Option 2: Fill NaN values with a default value (uncomment if needed)
        # test_df['label'] = test_df['label'].fillna(0)

    # Convert the label column to integers
    test_df['label'] = test_df['label'].astype(int)

    # Proceed with scoring
    # print('test acc: ', classifier.score(X_test, test_df['label']))

    # Tokenize words and calculate weights
    tokened_words = clip.tokenize(word_list).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    weights_model = LinearRegression(fit_intercept=False)
    weights_model.fit(word_features.cpu().T, classifier.coef_[0])
    word_df = pd.DataFrame({'word': words, 'weights': weights_model.coef_})
    word_df.sort_values('weights', inplace=True)
    word_df.set_index('word', inplace=True)
    word_df.to_csv(os.path.join(save_dir, f'word_weights-{save_tag}.csv'))

    # Predict probabilities
    yhat = classifier.predict_proba(X_test)
    X_test = X_test[:min(len(X_test), len(test_df.label))]
    test_df = test_df.iloc[:len(X_test)]
    print('test acc: ', classifier.score(X_test, test_df.label))

    pred_coef = weights_model.predict(word_features.cpu().T)
    cos_sim = 1 - distance.cosine(pred_coef, classifier.coef_[0])
    print('cosine sim between weights', cos_sim)


def get_prototypes(df, words, device, save_dir, n_save=20):
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    X = create_clip_feature_mat(df.file_path.values, clip_model, preprocess_fxn)

    tokened_words = clip.tokenize(words).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    file_dot = np.zeros((len(df), len(words)))
    for i in range(len(df)):
        for j in range(len(words)):
            file_dot[i, j] = np.dot(X[i], word_features[j].cpu())

    file_dot_pred = np.zeros((len(df), len(words)))
    for j in range(len(words)):
        fit_j = [k for k in range(len(words)) if k != j]
        dot_regression = LinearRegression()
        dot_regression.fit(file_dot[:, fit_j], file_dot[:, j])
        file_dot_pred[:, j] = dot_regression.predict(file_dot[:, fit_j])

    dot_df_diff = pd.DataFrame(file_dot - file_dot_pred, columns=words)
    dot_df_diff['label'] = df['label'].values
    dot_df_diff.set_index(df.file_path, inplace=True)

    for w in words:
        print(w)
        for sort_dir in ['top']:
            this_df = dot_df_diff.sort_values(w, ascending=(sort_dir == 'bottom'))
            save_files = this_df.index.values[:n_save]
            these_labels = this_df.label.values[:n_save]
            this_out_dir = save_dir + w + '_' + sort_dir + '/'
            if not os.path.exists(this_out_dir):
                os.mkdir(this_out_dir)

            for i, f in enumerate(save_files):
                if '.dcm' in f:
                    im = process_dcm(f)
                else:
                    im = imageio.imread(f)
                # make square and downsample for efficiency (CLIP also crops to square)
                min_dim = min(im.shape[:2])
                for dim in [0, 1]:
                    if im.shape[dim] > min_dim:
                        n_start = int((im.shape[dim] - min_dim) / 2)
                        n_stop = n_start + min_dim
                        if dim == 0:
                            # im = im[n_start:n_stop, :, :]
                            if len(im.shape) == 2:  # Grayscale image
                                im = im[n_start:n_stop, :]
                            elif len(im.shape) == 3:  # Color image
                                im = im[n_start:n_stop, :, :]
                            else:
                                raise ValueError(f"Unexpected image dimensions: {im.shape}")
                        else:
                            im = im[:, n_start:n_stop, :]
                if min_dim > 500:
                    im = cv2.resize(im, (500, 500))
                f_name = f'rank{i}_label{these_labels[i]}.png'
                imageio.imwrite(os.path.join(this_out_dir, f_name), im)

def convert_cbis_to_format(input_csv_path, output_csv_path):
    # input_csv_path = './data/csv/mass_case_description_test_set.csv'  # Input CSV file
    # output_csv_path = './data/cbis_mass_test.csv'  # Output CSV file

    # Read the input CSV
    df = pd.read_csv(input_csv_path)

    # df = df.head(5)

    def process_file_path(file_path):
        print(f"Processing file path: {file_path}")  # Debug: Print the input file path
        parts = file_path.split('/')
        if len(parts) > 1:
            folder_name = parts[2]  # First folder after /jpeg
            folder_path = os.path.join('./data/jpeg', folder_name)  # Add ./data prefix
            print(f"Looking in folder: {folder_path}")  # Debug: Print the folder path

            # Find the first .jpg file in the folder
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                print(f"Found JPG files: {jpg_files}")  # Debug: Print found .jpg files
                if jpg_files:
                    return os.path.join(folder_path, jpg_files[0])

        print("No valid .jpg file found")  # Debug: No .jpg file found
        return None

    # Extract relevant columns and map pathology to binary labels
    df_formatted = pd.DataFrame({
        'file_path': df['image file path'].apply(process_file_path),
        'label': df['pathology'].map({'MALIGNANT': 1, 'BENIGN': 0})
    })

    # Drop rows where file_path is None (no .jpg file found)
    df_formatted.dropna(subset=['file_path'], inplace=True)

    # Save the formatted DataFrame to a new CSV file
    df_formatted.to_csv(output_csv_path, index=False)
    print(f"Formatted CSV saved to {output_csv_path}")


if __name__ == '__main__':
    dataset_name = 'cbis'
    # device = 'cuda:0'
    device = 'cpu'

    print("right before cbis")
    # assumes a csv with columns containing file_path and label
    if dataset_name == 'cbis':
        convert_cbis_to_format('./data/csv/mass_case_description_test_set.csv', './data/cbis_mass_test.csv')
        convert_cbis_to_format('./data/csv/mass_case_description_train_set.csv', './data/cbis_mass_train.csv')
        train_path = './data/cbis_mass_train.csv'
        test_path = './data/cbis_mass_test.csv'
    elif dataset_name == 'melanoma':
        train_path = './data/siim_melanoma_train.csv'
        test_path = './data/siim_melanoma_test.csv'

    words = [
        'dark', 'light',
        'round', 'pointed',
        'large', 'small',
        'smooth', 'coarse',
        'transparent', 'opaque',
        'symmetric', 'asymmetric',
        'high contrast', 'low contrast',
        # New words
        'flat', 'bulging',
        'narrow', 'wide', 
        'gritty', 'sleek'
    ]

    base_out_dir = './results/'
    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    save_tag = dataset_name
    save_dir = base_out_dir + save_tag + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fit_words(train_df, test_df, device, words, save_dir=save_dir, save_tag=save_tag)

    prot_save_dir = os.path.join(save_dir, save_tag + '_prototypes/')
    if not os.path.exists(prot_save_dir):
        os.mkdir(prot_save_dir)
    get_prototypes(train_df, words, device, prot_save_dir, n_save=5)
