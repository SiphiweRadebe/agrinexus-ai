"""
Training Loop

Implements training and validation loops for the transformer model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
import time


class Trainer:
    """Training manager for the model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        loss_fn: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train.
            optimizer: Optimizer instance.
            device: Device to train on.
            loss_fn: Loss function (default: CrossEntropyLoss).
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
        
        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Reshape for loss computation
            logits_flat = logits.view(-1, self.model.vocab_size)
            labels_flat = labels.view(-1)
            
            # Compute loss
            loss = self.loss_fn(logits_flat, labels_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Batch {batch_idx + 1}: "
                    f"Loss = {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Reshape for loss computation
                logits_flat = logits.view(-1, self.model.vocab_size)
                labels_flat = labels.view(-1)
                
                # Compute loss
                loss = self.loss_fn(logits_flat, labels_flat)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_path: Optional[str] = None,
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            save_path: Path to save best model.
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            elapsed_time = time.time() - start_time
            
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
