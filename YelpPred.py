import json
import random
import pandas as pd
import re
import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from nltk.corpus import stopwords
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from collections import Counter
from sklearn.metrics import classification_report

# Constants
MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 10000
N_SAMPLES = 80000
NUM_CLASSES = 5
SAVE_DIR = "./model"

# Clean the text data
def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    # text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def load_data(n_samples=10000, file_path='yelp_academic_dataset_review.json'):
    random.seed(42)
    num_lines = sum(1 for l in open(file_path))
    keep_idx = set(random.sample(range(num_lines), n_samples))
    data = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i in keep_idx:
                data.append(json.loads(line))
    df = pd.DataFrame(data)
    df = df.drop(['review_id', 'user_id', 'business_id', 'date'], axis=1)
    df['text'] = df['text'].apply(clean_text)
    return df

# Function to remove top-k most frequent words from a specific dataset
def remove_top_k_words_from_texts(texts, k=10):
    all_words = ' '.join(texts).split()
    most_common_words = [word for word, count in Counter(all_words).most_common(k)]
    
    def clean_text(text):
        return ' '.join([word for word in text.split() if word not in most_common_words])
    
    cleaned_texts = [clean_text(text) for text in texts]
    return cleaned_texts

def data_preprocessing(n_samples, remove_low_ratings_flag=False, remove_top_k_words_flag=False):
    df = load_data(n_samples)
    df['stars'] = df['stars'].astype(int)

    # Drop any missing values
    df = df.dropna()
    df['stars'] -= 1  # Convert 1-5 to 0-4

    # Split the data first
    X_train_texts, X_temp_texts, y_train_stars, y_temp_stars = train_test_split(df['text'].values, df['stars'].values, test_size=0.4, random_state=42)
    X_valid_texts, X_test_texts, y_valid_stars, y_test_stars = train_test_split(X_temp_texts, y_temp_stars, test_size=0.5, random_state=42)

    # Further split for regression targets
    y_train_funny, y_temp_funny = train_test_split(df['funny'].values, test_size=0.4, random_state=42)
    y_valid_funny, y_test_funny = train_test_split(y_temp_funny, test_size=0.5, random_state=42)

    y_train_useful, y_temp_useful = train_test_split(df['useful'].values, test_size=0.4, random_state=42)
    y_valid_useful, y_test_useful = train_test_split(y_temp_useful, test_size=0.5, random_state=42)

    y_train_cool, y_temp_cool = train_test_split(df['cool'].values, test_size=0.4, random_state=42)
    y_valid_cool, y_test_cool = train_test_split(y_temp_cool, test_size=0.5, random_state=42)

    # Remove low ratings from y_train
    if remove_low_ratings_flag:
        mask = y_train_stars != 0  # 0 corresponds to 1-star rating after subtraction
        y_train_stars = y_train_stars[mask]

    # Remove top K most common words from X_train
    if remove_top_k_words_flag:
        X_train_texts = remove_top_k_words_from_texts(X_train_texts, k=10)

    # Prepare the texts and the labels
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(df['text']), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    def preprocess_batch(batch_texts, batch_stars, batch_funny, batch_useful, batch_cool):
        token_ids = [text_pipeline(text) for text in batch_texts]
        token_ids = [item[:MAX_SEQUENCE_LENGTH] for item in token_ids]
        token_ids = [pad(item) for item in token_ids]
        return (
            torch.tensor(token_ids, dtype=torch.long), 
            torch.tensor(batch_stars, dtype=torch.long),
            torch.tensor(batch_funny, dtype=torch.float),
            torch.tensor(batch_useful, dtype=torch.float),
            torch.tensor(batch_cool, dtype=torch.float)
        )

    def pad(sequence):
        return sequence + [0] * (MAX_SEQUENCE_LENGTH - len(sequence))

    X_train, y_train_stars, y_train_funny, y_train_useful, y_train_cool = preprocess_batch(
        X_train_texts, y_train_stars, y_train_funny, y_train_useful, y_train_cool)
    X_valid, y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool = preprocess_batch(
        X_valid_texts, y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool)
    X_test, y_test_stars, y_test_funny, y_test_useful, y_test_cool = preprocess_batch(
        X_test_texts, y_test_stars, y_test_funny, y_test_useful, y_test_cool)

    return X_train, X_valid, X_test, (y_train_stars, y_train_funny, y_train_useful, y_train_cool), (
        y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool), (y_test_stars, y_test_funny, y_test_useful, y_test_cool), vocab

class YelpReviewDataset(Dataset):
    def __init__(self, texts, stars, funny, useful, cool):
        self.texts = texts
        self.stars = stars
        self.funny = funny
        self.useful = useful
        self.cool = cool

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.stars[idx], self.funny[idx], self.useful[idx], self.cool[idx]

def create_data_loader(texts, stars, funny, useful, cool, batch_size):
    dataset = YelpReviewDataset(texts, stars, funny, useful, cool)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class MultiTaskCNNTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_filters, filter_sizes, dropout_rate):
        super(MultiTaskCNNTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(num_filters) for _ in filter_sizes])
        self.dropout = nn.Dropout(dropout_rate)  # Updated line to accept dropout rate
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.regressor = nn.Linear(len(filter_sizes) * num_filters, 3)  # 3 regression targets: funny, useful, cool

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add a channel dimension: (batch_size, 1, sequence_length, embed_size)
        x = [self.batch_norms[i](F.relu(conv(x))).squeeze(3) for i, conv in enumerate(self.convs)]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        classification_logits = self.classifier(x)
        regression_outputs = self.regressor(x)
        return classification_logits, regression_outputs

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    total_classification_loss = 0
    total_regression_loss = 0
    total_correct = 0
    total = 0

    for texts, stars, funny, useful, cool in tqdm(data_loader, desc="Training"):
        texts = texts.to(device)
        stars = stars.to(device)
        funny = funny.to(device)
        useful = useful.to(device)
        cool = cool.to(device)

        optimizer.zero_grad()
        classification_logits, regression_outputs = model(texts)
        classification_loss = F.cross_entropy(classification_logits, stars)
        regression_loss = F.mse_loss(regression_outputs[:, 0], funny) + \
                          F.mse_loss(regression_outputs[:, 1], useful) + \
                          F.mse_loss(regression_outputs[:, 2], cool)
        
        loss = classification_loss + regression_loss
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_classification_loss += classification_loss.item() * texts.size(0)
        total_regression_loss += regression_loss.item() * texts.size(0)
        total_correct += (classification_logits.argmax(1) == stars).sum().item()
        total += stars.size(0)

    return total_correct / total, total_classification_loss / total, total_regression_loss / total

def eval_model(model, data_loader, device):
    model.eval()
    total_classification_loss = 0
    total_regression_loss = 0
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    mse_funny, mse_useful, mse_cool = 0, 0, 0
    mae_funny, mae_useful, mae_cool = 0, 0, 0

    with torch.no_grad():
        for texts, stars, funny, useful, cool in tqdm(data_loader, desc="Evaluating"):
            texts = texts.to(device)
            stars = stars.to(device)
            funny = funny.to(device)
            useful = useful.to(device)
            cool = cool.to(device)

            classification_logits, regression_outputs = model(texts)
            classification_loss = F.cross_entropy(classification_logits, stars)
            regression_loss_funny = F.mse_loss(regression_outputs[:, 0], funny)
            regression_loss_useful = F.mse_loss(regression_outputs[:, 1], useful)
            regression_loss_cool = F.mse_loss(regression_outputs[:, 2], cool)

            regression_loss = regression_loss_funny + regression_loss_useful + regression_loss_cool

            total_classification_loss += classification_loss.item() * texts.size(0)
            total_regression_loss += regression_loss.item() * texts.size(0)
            total_correct += (classification_logits.argmax(1) == stars).sum().item()
            total += stars.size(0)

            all_preds.extend(classification_logits.argmax(1).cpu().numpy())
            all_labels.extend(stars.cpu().numpy())

            mse_funny += regression_loss_funny.item() * texts.size(0)
            mse_useful += regression_loss_useful.item() * texts.size(0)
            mse_cool += regression_loss_cool.item() * texts.size(0)

            mae_funny += F.l1_loss(regression_outputs[:, 0], funny).item() * texts.size(0)
            mae_useful += F.l1_loss(regression_outputs[:, 1], useful).item() * texts.size(0)
            mae_cool += F.l1_loss(regression_outputs[:, 2], cool).item() * texts.size(0)

    classification_accuracy = total_correct / total
    report = classification_report(all_labels, all_preds)
    avg_classification_loss = total_classification_loss / total
    avg_regression_loss = total_regression_loss / total
    mse_funny /= total
    mse_useful /= total
    mse_cool /= total
    rmse_funny = mse_funny ** 0.5
    rmse_useful = mse_useful ** 0.5
    rmse_cool = mse_cool ** 0.5
    mae_funny /= total
    mae_useful /= total
    mae_cool /= total

    return (classification_accuracy, report, avg_classification_loss, avg_regression_loss,
            mse_funny, mse_useful, mse_cool, rmse_funny, rmse_useful, rmse_cool, mae_funny, mae_useful, mae_cool)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), 'checkpoint.pt')

early_stopping = EarlyStopping(patience=5, delta=0.01)

def random_search(params, n_iter=10):
    """Random search for hyperparameter tuning."""
    keys = list(params.keys())
    best_score = None
    best_params = None
    
    for _ in range(n_iter):
        # Randomly sample a configuration
        current_params = {k: random.choice(v) for k, v in params.items()}
        print(f"Trying configuration: {current_params}")

        # Train and evaluate the model with the current configuration
        val_score = train_and_evaluate(current_params, use_small_subset=use_small_subset)

        if best_score is None or val_score < best_score:  # Want to minimize loss
            best_score = val_score
            best_params = current_params

    print(f"Best configuration: {best_params}")
    print(f"Best validation score: {best_score}")
    return best_params

def train_and_evaluate(params, use_small_subset=False):
    """Train and evaluate the model with the given parameters, optionally using a small subset for speed."""
    global EMBED_SIZE, NUM_FILTERS, FILTER_SIZES, DROPOUT_RATE, BATCH_SIZE
    EMBED_SIZE = params['embed_size']
    NUM_FILTERS = params['num_filters']
    FILTER_SIZES = params['filter_sizes']
    DROPOUT_RATE = params['dropout_rate']
    LEARNING_RATE = params['learning_rate']
    WEIGHT_DECAY = params['weight_decay']
    BATCH_SIZE = params['batch_size']
    
    # print("train_and_evaluate: use_small_subset: ", use_small_subset)
    if use_small_subset:
        X_train, X_valid, _, y_train, y_valid, _, vocab = data_preprocessing(5000)
    else:
        X_train, X_valid, X_test, y_train, y_valid, y_test, vocab = data_preprocessing(N_SAMPLES)
    
    train_loader = create_data_loader(X_train, y_train_stars, y_train_funny, y_train_useful, y_train_cool, BATCH_SIZE)
    valid_loader = create_data_loader(X_valid, y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskCNNTextModel(vocab_size=len(vocab), embed_size=EMBED_SIZE, num_classes=NUM_CLASSES,
                                  num_filters=NUM_FILTERS, filter_sizes=FILTER_SIZES, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    EPOCHS = 5 if use_small_subset else 20  # Use fewer epochs for small subset
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print('-' * 10)
        train_acc, train_class_loss, train_reg_loss = train_epoch(model, train_loader, optimizer, device, scheduler=None)
        print(f"Train classification loss: {train_class_loss:.4f}, Train regression loss: {train_reg_loss:.4f}, Train accuracy: {train_acc:.4f}")

        (val_acc, val_report, val_class_loss, val_reg_loss,
            mse_funny, mse_useful, mse_cool,
            rmse_funny, rmse_useful, rmse_cool, mae_funny, mae_useful, mae_cool) = eval_model(model, valid_loader, device)
        
        print(f"Validation classification loss: {val_class_loss:.4f}, Validation regression loss: {val_reg_loss:.4f}, Validation accuracy: {val_acc:.4f}")
        print(f"Validation MSE - Funny: {mse_funny:.4f}, Useful: {mse_useful:.4f}, Cool: {mse_cool:.4f}")
        print(f"Validation RMSE - Funny: {rmse_funny:.4f}, Useful: {rmse_useful:.4f}, Cool: {rmse_cool:.4f}")
        print(f"Validation MAE - Funny: {mae_funny:.4f}, Useful: {mae_useful:.4f}, Cool: {mae_cool:.4f}")

        val_loss = val_class_loss + val_reg_loss
        scheduler.step(val_loss)

        early_stopping(val_loss, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return best_val_loss  # Return validation score for model selection

def save_params(params, filepath="best_params.json"):
    """Save parameters to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(params, f)

def save_model(model, filepath="best_model.pth"):
    """Save the model state to a file."""
    torch.save(model.state_dict(), filepath)


def load_params(filepath="best_params.json"):
    """Load parameters from a JSON file, if it exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        return params
    return None

def load_model(model, filepath="best_model.pth"):
    """Load the model state from a file, if it exists."""
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        return model
    return None

if __name__ == '__main__':
    global remove_low_ratings_flag, remove_top_k_words_flag, train_model, fine_tuning, use_small_subset
    remove_low_ratings_flag = True
    remove_top_k_words_flag = False
    train_model = True
    fine_tuning = True
    use_small_subset = True
    
    if len(sys.argv) > 1:
        remove_low_ratings_flag = sys.argv[1]
    if len(sys.argv) > 2:
        remove_top_k_words_flag = sys.argv[2]
    if len(sys.argv) > 3:
        train_model = sys.argv[3]
    if len(sys.argv) > 4:
        fine_tuning = sys.argv[4]
    if len(sys.argv) > 5:
        use_small_subset = sys.argv[5]

    best_params_filepath = "best_params.json"
    if remove_low_ratings_flag:
        best_model_filepath = "best_model_remove_low_ratings.pth"
    elif remove_top_k_words_flag:
        best_model_filepath = "best_model_remove_top_k.pth"
    else:
        best_model_filepath = "best_model.pth"

    # Load best parameters or perform random search
    best_params = load_params(filepath=best_params_filepath)

    # Prepare data loaders
    X_train, X_valid, X_test, y_train, y_valid, y_test, vocab = data_preprocessing(N_SAMPLES)
    y_train_stars, y_train_funny, y_train_useful, y_train_cool = y_train
    y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool = y_valid
    y_test_stars, y_test_funny, y_test_useful, y_test_cool = y_test

    print("Fine-tuning: ", fine_tuning)
    if fine_tuning:
        if best_params is None:
            start_time = time.time()

            # Define the parameter grid for random search
            param_grid = {
                "embed_size": [50, 100, 200],
                "num_filters": [50, 100, 150],
                "filter_sizes": [[3, 4, 5], [2, 3, 4, 5], [3, 5, 7]],
                "dropout_rate": [0.3, 0.5, 0.7],
                "learning_rate": [1e-3, 1e-4, 1e-5],
                "weight_decay": [1e-5, 1e-6, 1e-7],
                "batch_size": [32, 64, 128]
            }

            # Perform random search
            best_params = random_search(param_grid, n_iter=10)

            # Save the best parameters
            save_params(best_params, filepath=best_params_filepath)

            end_time = time.time()
            print(f"Random search completed in {end_time - start_time:.2f} seconds.")

        # Use best params to define and train final model
        global EMBED_SIZE, NUM_FILTERS, FILTER_SIZES, DROPOUT_RATE, BATCH_SIZE
        EMBED_SIZE = best_params['embed_size']
        NUM_FILTERS = best_params['num_filters']
        FILTER_SIZES = best_params['filter_sizes']
        DROPOUT_RATE = best_params['dropout_rate']
        LEARNING_RATE = best_params['learning_rate']
        WEIGHT_DECAY = best_params['weight_decay']
        BATCH_SIZE = best_params['batch_size']
    else:
        # Default parameters
        BATCH_SIZE = 32
        EMBED_SIZE = 100
        NUM_CLASSES = 5
        NUM_FILTERS = 100
        FILTER_SIZES = [3, 4, 5]
        DROPOUT_RATE = 0.5
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = 1e-5
    
    train_loader = create_data_loader(X_train, y_train_stars, y_train_funny, y_train_useful, y_train_cool, BATCH_SIZE)
    valid_loader = create_data_loader(X_valid, y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool, BATCH_SIZE)
    test_loader = create_data_loader(X_test, y_test_stars, y_test_funny, y_test_useful, y_test_cool, BATCH_SIZE)

    # print("cuda available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskCNNTextModel(vocab_size=len(vocab), embed_size=EMBED_SIZE, num_classes=NUM_CLASSES, num_filters=NUM_FILTERS, filter_sizes=FILTER_SIZES, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    # Load model if it exists, otherwise train and save it
    if not os.path.exists(best_model_filepath) or train_model:
        start_time = time.time()

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        EPOCHS = 20
        early_stopping = EarlyStopping(patience=5, delta=0.01)

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print('-' * 10)
            train_acc, train_class_loss, train_reg_loss = train_epoch(model, train_loader, optimizer, device, scheduler=None)
            print(f"Train classification loss: {train_class_loss:.4f}, Train regression loss: {train_reg_loss:.4f}, Train accuracy: {train_acc:.4f}")

            (val_acc, val_report, val_class_loss, val_reg_loss,
                mse_funny, mse_useful, mse_cool,
                rmse_funny, rmse_useful, rmse_cool, mae_funny, mae_useful, mae_cool) = eval_model(model, valid_loader, device)
            
            print(f"Validation classification loss: {val_class_loss:.4f}, Validation regression loss: {val_reg_loss:.4f}, Validation accuracy: {val_acc:.4f}")
            print(f"Validation MSE - Funny: {mse_funny:.4f}, Useful: {mse_useful:.4f}, Cool: {mse_cool:.4f}")
            print(f"Validation RMSE - Funny: {rmse_funny:.4f}, Useful: {rmse_useful:.4f}, Cool: {rmse_cool:.4f}")
            print(f"Validation MAE - Funny: {mae_funny:.4f}, Useful: {mae_useful:.4f}, Cool: {mae_cool:.4f}")

            val_loss = val_class_loss + val_reg_loss
            scheduler.step(val_loss)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Training complete. Saving the model.")
        save_model(model, filepath=best_model_filepath)

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
    else:
        model = load_model(model, filepath=best_model_filepath)
        if model:
            print("Loaded saved model.")

    # Evaluate the final model on the test set
    (test_acc, test_report, test_class_loss, test_reg_loss,
     mse_funny, mse_useful, mse_cool,
     rmse_funny, rmse_useful, rmse_cool, mae_funny, mae_useful, mae_cool) = eval_model(model, test_loader, device)

    print(f"Test classification loss: {test_class_loss:.4f}, Test regression loss: {test_reg_loss:.4f}, Test accuracy: {test_acc:.4f}", "\n", test_report)
    print(f"Test MSE - Funny: {mse_funny:.4f}, Useful: {mse_useful:.4f}, Cool: {mse_cool:.4f}")
    print(f"Test RMSE - Funny: {rmse_funny:.4f}, Useful: {rmse_useful:.4f}, Cool: {rmse_cool:.4f}")
    print(f"Test MAE - Funny: {mae_funny:.4f}, Useful: {mae_useful:.4f}, Cool: {mae_cool:.4f}")