import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 读取加载文件
def load_data(path, auto_max_length=False):
    texts, labels = [], []
    encodings = ['utf-8', 'utf-16-le', 'utf-16-be', 'gb18030', 'big5']
    max_text_length = 0  # 用于存储数据集中最长文本的长度

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    for line in f:
                        label, text = line.split(" ", 1)
                        texts.append(text.strip())
                        labels.append(int(label))
                        if auto_max_length:  # 如果启用自动计算 max_length，则需要计算每个文本的长度
                            max_text_length = max(max_text_length, len(text.strip()))
                break  # 如果文件成功以某种编码打开并读取，就跳出循环，处理下一个文件
            except UnicodeError:
                continue  # 如果以某种编码打开文件失败，就尝试下一种编码
        else:
            print(f"Cannot open file {file} at path {file_path} with any of the provided encodings.")
    return texts, labels, max_text_length


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class SentimentClassifier(nn.Module):
    def __init__(self, transformer_model, hidden_dim, output_dim, dropout, use_transformer, nhead, num_transformer_layers, num_lstm_layers, vocab_size=None):
        super(SentimentClassifier, self).__init__()
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = AutoModel.from_pretrained(transformer_model)
            input_dim = self.transformer.config.hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
            self.transformer = TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
            input_dim = hidden_dim
            
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        if self.use_transformer:
            transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
            hidden_states = transformer_outputs.last_hidden_state
        else:
            embedded = self.embedding(input_ids)
            hidden_states = self.transformer(embedded)

        lstm_out, _ = self.lstm(hidden_states)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return torch.sigmoid(out)


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        input_ids, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in dataloader:
            input_ids, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item()
            preds = outputs.squeeze().cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

    y_pred_labels = [1 if pred > 0.5 else 0 for pred in y_pred] 
    accuracy = accuracy_score(y_true, y_pred_labels)  
    report = classification_report(y_true, y_pred_labels, digits=4) 
    return running_loss / len(dataloader), accuracy, report


# 超参数设置
num_lstm_layers = 2  # LSTM的层数
hidden_dim = 128 # 隐藏层维度。
output_dim = 1 # 输出层维度，2分类为问题维度为1。
dropout = 0.3 # 神经网络中 dropout 的比例。
batch_size = 64 # 每个批次的样本数量。
num_epochs = 100 # 整个数据集的训练次数。

# 数据集最大长度截断
auto_max_length = True # 是否自动根据数据集调整max_length。当小于128时使用数据集最大长度，当大于等于128小于512时使用数据集最大长度的80%，当大于512时使用512。
max_length = 128 # 输入文本的最大长度，超过此长度的文本将被截断。

# 弦退火学习率
use_cosine_annealing = True # 是否使用余弦退火学习率调整策略。如果为 True，则使用余弦退火策略；否则使用固定学习率。
initial_learning_rate = 0.001 # 初始学习率，当为启用弦退火学习率时为固定学习率。
min_learning_rate = 0.0001 # 余弦退火学习率的最小值。

# transformer设置
use_transformer = True # 是否使用预训练的 transformer 模型。如果为 True，则使用预训练的 transformer 模型；否则使用自定义的 transformer 编码器。
nhead = 12 # Transformer 编码器中自注意力机制的头数。
num_transformer_layers = 12 # Transformer 编码器的层数。

# 数据集和模型设置
data_path = "path/to/your/data" # 包含数据集的文件夹路径。
model_save_path = "path/to/your/models"  # 用于保存模型的文件夹路径。
transformer_model = "chinese-lert-base" # 预训练的 transformer 模型的名称，这里使用的是哈工大的lert，更多模型可以去huggingface上寻找。
vocab_size = 30522  # 词汇表大小。如果不使用预训练的 transformer 模型，需要设置为正确的词汇表大小。


def main():
    texts, labels, max_text_length = load_data(data_path, auto_max_length)
    if auto_max_length:
        if max_text_length < 128:
            max_length = max_text_length
        elif 128 <= max_text_length < 512:
            max_length = int(max_text_length * 0.8)
        else:
            max_length = 512
    else:
        max_length = 128
        
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentClassifier(transformer_model, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, 
                                use_transformer=use_transformer, nhead=nhead, num_transformer_layers=num_transformer_layers, 
                                num_lstm_layers=num_lstm_layers,
                                vocab_size=vocab_size if not use_transformer else None).to(device)

    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    criterion = nn.BCELoss()

    # 计算每个 epoch 的迭代次数
    if use_cosine_annealing:
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
        # 使用 num_epochs * len(train_loader) 作为 T_max 参数
        total_iterations = num_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=min_learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)


    best_val_loss = float('inf')

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler if use_cosine_annealing else None)
        val_loss, val_accuracy, report = evaluate(model, val_loader, criterion, device)
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(report)

        # 每 10 个 epoch 保存一次模型，如果当前模型更好，则更新最佳模型.
        if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
            os.makedirs(model_save_path, exist_ok=True)  # 确保保存路径存在
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{epoch+1}.pt'))  # 保存模型
            if val_accuracy > best_accuracy:  # 如果当前模型效果比上一个更好
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pt'))  # 保存最佳模型


if __name__ == "__main__":
    main()
