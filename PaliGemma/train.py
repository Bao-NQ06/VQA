import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration

import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description='Train VQA Model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def setup_logging(checkpoint_dir):
    logging.basicConfig(
        filename=f'{checkpoint_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, (images, questions, answers) in enumerate(pbar):
                # Move data to device
                images = images.to(device)
                questions = questions.to(device)
                answers = answers.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images, questions)
                loss = criterion(outputs, answers)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, questions, answers in val_loader:
                images = images.to(device)
                questions = questions.to(device)
                answers = answers.to(device)
                
                outputs = model(images, questions)
                loss = criterion(outputs, answers)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
    return model

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Initialize model (assuming it's defined elsewhere)
    model = PaliGemmaForConditionalGeneration().to(device)
    
    # Setup data loaders (assuming you have dataset classes defined)
    train_dataset = YourDataset(split='train')
    val_dataset = YourDataset(split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device
    )

if __name__ == '__main__':
    main()
