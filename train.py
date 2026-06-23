"""
Training Script

Entry point for training the AgriNexus AI model.
Usage: python train.py --epochs 10 --batch-size 32 --learning-rate 0.001
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.model import TransformerModel
from src.tokenizer import CharTokenizer
from src.training import Trainer
from src.utils import setup_device, count_parameters


def create_dummy_dataloaders(batch_size: int = 32, num_batches: int = 100):
    """
    Create dummy data loaders for testing.
    
    TODO: Replace with actual data loading from agricultural datasets.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Generate random dummy data
    input_ids = torch.randint(0, 100, (num_batches * batch_size, 50))
    labels = torch.randint(0, 100, (num_batches * batch_size, 50))
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([b[0] for b in batch]),
            'labels': torch.stack([b[1] for b in batch]),
        }
    )
    
    return dataloader, dataloader


def main(args):
    """Main training function."""
    
    # Setup
    print("🌾 AgriNexus AI - Training Script")
    print("=" * 50)
    
    device = setup_device(use_cuda=not args.cpu)
    
    # Create output directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print("\n📝 Initializing tokenizer...")
    tokenizer = CharTokenizer()
    # TODO: Build from actual agricultural dataset
    tokenizer.vocab = {
        f"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
        **{chr(i): i + 4 for i in range(32, 127)}  # ASCII characters
    }
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    
    print(f"✅ Tokenizer ready (vocab size: {tokenizer.get_vocab_size()})")
    
    # Initialize model
    print("\n🧠 Initializing transformer model...")
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    
    num_params = count_parameters(model)
    print(f"✅ Model created with {num_params:,} trainable parameters")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
    )
    
    # Create data loaders
    print("\n📊 Preparing data...")
    train_loader, val_loader = create_dummy_dataloaders(
        batch_size=args.batch_size,
    )
    print(f"✅ Data loaders ready")
    
    # Train
    print("\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print("-" * 50)
    
    save_path = checkpoint_dir / "best_model.pt"
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=str(save_path),
    )
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    trainer.save_checkpoint(str(final_path))
    
    print("\n" + "=" * 50)
    print("✅ Training complete!")
    print(f"📦 Models saved to {checkpoint_dir}")
    
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AgriNexus AI model"
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay (L2 regularization)")
    
    # Model hyperparameters
    parser.add_argument("--d-model", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1024,
                        help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # System
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    args = parser.parse_args()
    main(args)
