# Spam SMS Detection with Multinomial Naive Bayes

This project implements a machine learning model to classify SMS messages as either "spam" or "ham" (not spam) using the Multinomial Naive Bayes algorithm. The code is provided in a Google Colab/Jupyter notebook and covers the entire process from data loading and preprocessing to model training, evaluation, and prediction.

## Project Overview

The primary objective of this project is to build an effective classifier that can distinguish between legitimate SMS messages and unsolicited spam messages. We utilize a standard dataset for SMS spam collection and employ the Multinomial Naive Bayes algorithm, which is a common and often effective choice for text classification tasks, especially with features like TF-IDF.

The workflow includes:

*   Loading and initial exploration of the dataset.
*   Cleaning and preparing the text data for machine learning.
*   Transforming text into numerical features using TF-IDF.
*   Splitting the data into training and testing sets.
*   Training the Multinomial Naive Bayes model.
*   Evaluating the model's performance using standard classification metrics.
*   Providing a function to make predictions on new, unseen messages.

## Dataset

The project uses the "SMS Spam Collection Dataset". This dataset is typically distributed as a `.csv` file (often named `spam.csv`) and contains a collection of SMS messages along with their labels ("ham" or "spam").

*   **Source:** This dataset is widely available online and is a common benchmark for spam filtering. (Note: Due to repository guidelines, a direct link is not provided, but searching for "SMS Spam Collection Dataset" will yield results).
*   **Structure:** The original dataset often includes unnecessary columns. The code handles this by dropping columns named 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4', and renames the relevant columns to 'label' and 'message'.
*   **Labels:** The 'label' column contains either 'ham' or 'spam'. These are converted to numerical labels (0 for ham, 1 for spam) for model training.

## Requirements

Ensure you have the following Python libraries installed in your environment. You can install them using pip:

Specifically, the notebook uses:

*   `pandas` for data manipulation and analysis.
*   `numpy` for numerical operations.
*   `matplotlib` and `seaborn` for data visualization.
*   `scikit-learn` for machine learning functionalities (data splitting, TF-IDF, Naive Bayes, evaluation metrics).
*   `nltk` for potential future advanced text preprocessing (though it's commented out in the current version).

## Setup and Usage

1.  **Clone the repository:**
    If you're using Git, clone the repository to your local machine:

    Replace `<repository_url>` with the URL of this repository.

2.  **Download the dataset:**
    Obtain the `spam.csv` file for the SMS Spam Collection dataset. Place this file in the same directory where you will run the notebook. The notebook includes a `try-except` block to check for the file's existence.

3.  **Open and run the notebook:**
    The core of the project is in the Python notebook file (e.g., `spam_detection_notebook.ipynb`). Open this file using Google Colab, Jupyter Notebook, or JupyterLab. Execute the cells sequentially from top to bottom.

    *   **Google Colab:** Upload the notebook file and the `spam.csv` file to your Colab environment.
    *   **Jupyter Notebook/Lab:** Navigate to the directory containing the files and open the notebook.

## Code Explanation

The notebook is structured into logical sections, each addressing a specific part of the machine learning workflow:

*   **Import Libraries:** Imports all necessary Python libraries at the beginning.
*   **Load and Explore Data:**
    *   Reads the `spam.csv` file into a pandas DataFrame. Includes error handling for `FileNotFoundError`.
    *   Drops irrelevant columns ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4').
    *   Renames 'v1' to 'label' and 'v2' to 'message'.
    *   Prints the head of the DataFrame, its shape, info, and checks for null values.
    *   Shows the distribution of 'ham' and 'spam' labels.
    *   Converts the 'label' column to numerical representation ('label_num': 0 for ham, 1 for spam).
*   **Visualize Data:**
    *   Uses `seaborn.countplot` to visualize the distribution of 'ham' vs. 'spam' labels.
    *   Adds a 'message_length' column to the DataFrame by calculating the length of each message.
    *   Uses `seaborn.histplot` to visualize the distribution of message lengths for both ham and spam messages, which can reveal useful patterns (e.g., spam messages often being longer).
*   **Preprocessing:**
    *   Creates a new column 'processed_message' by converting all messages to lowercase. This is a basic but effective text normalization step.
    *   Includes a commented-out section demonstrating how to perform more advanced preprocessing using NLTK (tokenization, removing punctuation, removing stop words, stemming). This can be uncommented and used for experimentation.
    *   Selects 'processed_message' as the feature (`X_text`) and 'label_num' as the target variable (`y`).
*   **Split Data:**
    *   Uses `sklearn.model_selection.train_test_split` to divide the data into training (75%) and testing (25%) sets.
    *   `random_state=42` ensures reproducibility of the split.
    *   `stratify=y` ensures that the proportion of ham and spam messages is the same in both the training and testing sets, which is important for imbalanced datasets.
    *   Prints the sizes of the training and testing sets.
*   **Vectorization:**
    *   Initializes a `TfidfVectorizer` from `sklearn.feature_extraction.text`.
    *   `stop_words='english'` removes common English stop words during the vectorization process.
    *   `max_df=0.9` ignores terms that appear in more than 90% of the documents (useful for removing terms that are too common to be informative).
    *   `min_df=2` ignores terms that appear in fewer than 2 documents (useful for removing rare terms).
    *   Fits the vectorizer on the training data (`X_train_text`) using `fit_transform()`. This learns the vocabulary and IDF values from the training data.
    *   Transforms the test data (`X_test_text`) using `transform()`. It's crucial to only *transform* the test data using the vectorizer *fitted* on the training data to avoid data leakage.
    *   Prints the shape of the resulting TF-IDF matrices, showing the number of documents and the size of the vocabulary (number of unique terms).
*   **Train Model:**
    *   Initializes a `MultinomialNB` classifier from `sklearn.naive_bayes`.
    *   `alpha=1.0` is the smoothing parameter; a value of 1.0 is standard Laplace smoothing. This helps prevent zero probabilities for terms that don't appear in the training data.
    *   Trains the classifier using the TF-IDF vectorized training data (`X_train_tfidf`) and the training labels (`y_train`) using the `fit()` method.
    *   Includes a commented-out section for training a `LogisticRegression` model as an alternative classifier.
*   **Evaluate Model:**
    *   Defines a helper function `evaluate_model` to calculate and print common classification metrics:
        *   Accuracy
        *   Precision (specifically for the 'spam' class, `pos_label=1`)
        *   Recall (specifically for the 'spam' class)
        *   F1-Score (specifically for the 'spam' class)
        *   Confusion Matrix
        *   Classification Report (including precision, recall, f1-score for both classes)
    *   Calls `evaluate_model` to assess the performance of the trained Multinomial Naive Bayes classifier on the test data (`y_test`, `y_pred_nb`).
*   **Predict on New Messages:**
    *   Defines a function `predict_new_message` that takes a raw message, the fitted TF-IDF vectorizer, and the trained model as input.
    *   Applies the same preprocessing (lowercasing) to the new message.
    *   Uses the fitted vectorizer to transform the new message into the TF-IDF representation.
    *   Uses the trained model's `predict()` method to get the predicted class (0 or 1).
    *   Uses the model's `predict_proba()` method to get the probability of the message belonging to each class (ham and spam).
    *   Returns the predicted label ('ham' or 'spam') and the class probabilities.
    *   Provides example usage of the `predict_new_message` function with sample spam and ham messages.

## Results

After running the notebook, the "Evaluate Model" section will output the performance metrics for the Multinomial Naive Bayes classifier on the held-out test set. Pay attention to:

*   **Accuracy:** Overall percentage of correctly classified messages.
*   **Precision (Spam):** Out of all messages predicted as spam, what percentage were actually spam? (Important for minimizing false positives - classifying a ham message as spam).
*   **Recall (Spam):** Out of all actual spam messages, what percentage were correctly identified? (Important for minimizing false negatives - failing to detect a spam message).
*   **F1-Score (Spam):** The harmonic mean of Precision and Recall, providing a single metric that balances both.
*   **Confusion Matrix:** A table showing the counts of True Positives, True Negatives, False Positives, and False Negatives.
*   **Classification Report:** A comprehensive report showing precision, recall, and f1-score for both 'ham' and 'spam' classes.

The "Predict on New Messages" section demonstrates how the model performs on specific examples, showing the predicted class and the model's confidence (probabilities).

## Potential Improvements

*   **Advanced Preprocessing:** Implement and experiment with the commented-out NLTK-based preprocessing (tokenization, stop word removal, stemming/lemmatization).
*   **Hyperparameter Tuning:** Experiment with the hyperparameters of the `TfidfVectorizer` (e.g., `ngram_range`, `max_features`, `min_df`, `max_df`) and the `MultinomialNB` classifier (e.g., `alpha`). Grid search or random search can be used for this.
*   **Alternative Models:** Train and evaluate other text classification models like Logistic Regression, Support Vector Machines (SVM), or even deep learning models (e.g., Recurrent Neural Networks or Transformers) for comparison.
*   **Feature Engineering:** Explore other features beyond message length, such as the number of special characters, the presence of URLs, or the use of excessive capitalization.
*   **Handling Imbalance:** If the dataset were severely imbalanced (much more ham than spam), techniques like oversampling the minority class (spam) or undersampling the majority class (ham) could be considered during training.

## Contributing

Contributions are welcome! If you find a bug, want to add a feature, or improve the documentation, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Create a new Pull Request.
