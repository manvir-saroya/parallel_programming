''' Parallelizing Fine Tuning of a Pre-Trained Model for Sentiment Analysis'''

!git clone https://github.com/rapidsai/rapidsai-csp-utils.git      #Downloads the CUDA Rapids
!python rapidsai-csp-utils/colab/pip-install.py
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install transformers datasets pandas numpy matplotlib seaborn torch tqdm  # Download other Dependencies

#Remove "!" if executing as a .py , Otherwise Run on a Google Colab or Jupyter Notebook Environment with NVIDIA enabled GPU

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cudf
import cuml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import gc
import os
from datasets import load_dataset

# Times the Processes , CPU and GPU
class Timer:
    def __init__(self, name):
        self.name = name
        self.times = []

    def __enter__(self):
        if self.name == 'gpu':
            torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if self.name == 'gpu':
            torch.cuda.synchronize()
        self.times.append(time.time() - self.start)

# Dataset Loading from Hugging Face , Can load any dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Loading in Chunks to Maintain Efficient Memory Management and Prevent the RAM or the GPU from Crashing
def load_data_in_chunks(split='train', chunk_size=10000):  # Change Accordingly
    dataset = load_dataset("", split=split)  # Load Required Dataset
    chunks = []

    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i + chunk_size]
        df_chunk = pd.DataFrame({
            'text': chunk['text'],
            'label': chunk['label']
        })
        chunks.append(df_chunk)
        gc.collect()

    return pd.concat(chunks, ignore_index=True)

# Clears Memory to maintain Memory efficiency and remove any Redundant Features
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()

#CPU performs better in smaller Batch Sizes Also Try batch_size = 64 or 128
def process_data_cpu(df, tokenizer, batch_size=32):
    dataset = SentimentDataset(df['text'].values, df['label'].values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

#GPU can handle larger batch sizes and results in Speedup ( Change Batch Size Accordingly )
#Same Batch sizes for fair comparison
def process_data_gpu(df, tokenizer, batch_size=32):

    print("Pre-tokenizing data for GPU...")

    chunk_size = 10000
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for i in tqdm(range(0, len(df), chunk_size), desc="Processing chunks"):
        chunk_df = df.iloc[i:i + chunk_size]
        encodings = tokenizer(
            chunk_df['text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        all_input_ids.append(encodings['input_ids'].cuda())
        all_attention_masks.append(encodings['attention_mask'].cuda())
        all_labels.append(torch.tensor(chunk_df['label'].values, dtype=torch.long).cuda())

    gpu_input_ids = torch.cat(all_input_ids)
    gpu_attention_mask = torch.cat(all_attention_masks)
    gpu_labels = torch.cat(all_labels)

    dataset = GPUDataset(gpu_input_ids, gpu_attention_mask, gpu_labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0
    )

    return dataloader

# Training the Model using AdamW optimizer ( better for sentiment analysis tasks ) Can change accordingly
def train_model(model, dataloader, device, epochs=3, batch_size=32):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Clear memory after each batch
            del input_ids, attention_mask, labels, outputs, loss
            clear_memory()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        clear_memory()

    return model, losses

# Evaluation of the Models
def evaluate_model(model, dataloader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)


            del input_ids, attention_mask, labels, outputs, predictions
            clear_memory()

    return correct / total

# Plotting 
# Add More detailed Plotting As Required
def plot_comparison(results):
    plt.switch_backend('agg')


    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Training Time
    plt.subplot(2, 2, 1)
    times = [cpu_results['train_time'], gpu_results['train_time']]
    plt.bar(['CPU', 'GPU'], times)
    plt.title('Training Time')
    plt.ylabel('Seconds')

    # Plot 2: Training Loss Over Epochs
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(cpu_results['losses']) + 1), cpu_results['losses'],
            marker='o', label='CPU')
    if 'losses' in gpu_results:
        plt.plot(range(1, len(gpu_results['losses']) + 1), gpu_results['losses'],
                marker='o', label='GPU')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 3: Speedup
    plt.subplot(2, 2, 3)
    speedup = cpu_results['train_time'] / gpu_results['train_time']
    plt.bar(['GPU Speedup'], [speedup])
    plt.title('GPU Speedup (x times faster)')
    plt.ylabel('Speedup Factor')

    # Plot 4: GPU Accuracy
    plt.subplot(2, 2, 4)
    plt.bar(['GPU'], [gpu_results['accuracy']])
    plt.title('GPU Model Accuracy')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('gpu_vs_cpu_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset(batch_size=32):

    results = {}

    print("\nProcessing Tweet Sentiment Dataset...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Loading training data...")
    train_df = load_data_in_chunks('train')

    # CPU Pipeline
    print("\nRunning CPU pipeline...")
    with Timer('cpu') as t:
        train_dataloader_cpu = process_data_cpu(train_df, tokenizer, batch_size)
        cpu_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        cpu_model, cpu_losses = train_model(cpu_model, train_dataloader_cpu, 'cpu', batch_size=batch_size)
    results['cpu_train_time'] = t.times[-1]

    # Load and process test data
    print("Loading test data...")
    test_df = load_data_in_chunks('test')
    test_dataloader_cpu = process_data_cpu(test_df, tokenizer, batch_size)
    results['cpu_accuracy'] = evaluate_model(cpu_model, test_dataloader_cpu, 'cpu')
    cpu_results['losses'] = cpu_losses

    clear_memory()

    # GPU Pipeline
    if torch.cuda.is_available():
        print("\nRunning GPU pipeline...")
        with Timer('gpu') as t:
            train_dataloader_gpu = process_data_gpu(train_df, tokenizer, batch_size)
            gpu_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).cuda()
            gpu_model, gpu_losses = train_model(gpu_model, train_dataloader_gpu, 'cuda', batch_size=batch_size)
        results['gpu_train_time'] = t.times[-1]

        test_dataloader_gpu = process_data_gpu(test_df, tokenizer, batch_size)
        results['gpu_accuracy'] = evaluate_model(gpu_model, test_dataloader_gpu, 'cuda')
        gpu_results['losses'] = gpu_losses

    clear_memory()

    plot_comparison(results)

    return results

def main():

    BATCH_SIZE = 16  # Change it according to the requirements

    try:
        results = process_dataset(BATCH_SIZE)
        
        speedup = cpu_results['train_time'] / gpu_results['train_time']
        print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")
        
        print("\nPlotting Results saved as tweet_sentiment_results.png")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


    
    finally:
        clear_memory()

if __name__ == "__main__":
    main()


'''
If the pipeline does not work , try implementing this class for the GPU dataset
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GPUDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
'''
