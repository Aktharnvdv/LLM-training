import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import Dataset, DataLoader
import json

# Load training data from a JSON file
with open('train_data.json', 'r', encoding='utf-8') as json_file:
    train_data = json.load(json_file)

# Custom dataset class for training
class MyCustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Separate user and assistant examples based on their roles
        user_examples = [e["content"] for e in self.data if e["role"] == "user"]
        user_content = user_examples[idx % len(user_examples)]  

        assistant_examples = [e["content"] for e in self.data if e["role"] == "assistant"]
        assistant_content = assistant_examples[idx % len(assistant_examples)] 

        # Tokenize user and assistant inputs
        user_inputs = self.tokenizer(user_content, max_length=self.max_length, 
                                     padding="max_length", truncation=True, return_tensors="pt")
        user_inputs = {k: v.squeeze(0) for k, v in user_inputs.items()}

        assistant_inputs = self.tokenizer(assistant_content, max_length=self.max_length, 
                                          padding="max_length", truncation=True, return_tensors="pt")
        assistant_inputs = {k: v.squeeze(0) for k, v in assistant_inputs.items()}

        return {"user_inputs": user_inputs, "assistant_inputs": assistant_inputs}

# Training function for a single epoch
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):

        # Move inputs to the specified device
        user_inputs = {k: v.to(device) for k, v in batch["user_inputs"].items()}
        assistant_inputs = {k: v.to(device) for k, v in batch["assistant_inputs"].items()}

        # Forward pass through the model and compute loss
        outputs = model(**user_inputs, labels=assistant_inputs["input_ids"])
        loss = outputs.loss
        total_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print examples every 1000 batches
        if i % 1000 == 0:
            print("\nExample:")
            input_text = tokenizer.decode(batch["user_inputs"]["input_ids"][0], skip_special_tokens=True)
            print(f"Input Text: {input_text}")

            generated_ids = outputs.logits.argmax(dim=-1)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated Text: {generated_text}")

            gold_text = tokenizer.decode(assistant_inputs["input_ids"][0], skip_special_tokens=True)
            print(f"Gold Label (Assistant Input): {gold_text}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Set the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer configurations
model_name = "llmware/bling-falcon-1b-0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, revision="main", 
                                          auth_token="*******************************")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.to(device)

# AdamW optimizer for training
optimizer = AdamW(model.parameters(), lr=1e-5)

# Create an instance of the custom dataset and DataLoader for training
train_dataset = MyCustomDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Number of training epochs
num_epochs = 150000

# Training loop
for epoch in range(num_epochs):
    avg_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
