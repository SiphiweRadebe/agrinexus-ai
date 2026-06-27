"""
Training Script for Agricultural Data

Train AgriNexus AI on the agricultural Q&A dataset.
Usage: python train_on_agri_data.py --epochs 10 --batch-size 16
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from src.model import TransformerModel
from src.tokenizer import CharTokenizer
from src.training import Trainer
from src.data_loader import create_agricultural_dataloader, get_dataset_stats
from src.utils import setup_device, count_parameters


def main(args):
    """Main training function."""
    
    print("\n🌾 AgriNexus AI - Agricultural Data Training Script")
    print("=" * 60)
    
    # Setup
    device = setup_device(use_cuda=not args.cpu)
    
    # Create output directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print("\n📝 Initializing tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.vocab = {
        "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
        **{chr(i): i + 4 for i in range(32, 127)}
    }
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    print(f"✅ Tokenizer ready (vocab size: {tokenizer.get_vocab_size()})")
    
    # Load and display dataset
    dataset_path = "data/agricultural_qa_dataset.json"
    print("\n📊 Loading agricultural dataset...")
    stats = get_dataset_stats(dataset_path)
    print(f"✅ Dataset loaded:")
    print(f"   - Total Q&A pairs: {stats.get('total_qa_pairs', 0)}")
    print(f"   - Avg question length: {stats.get('avg_question_length_words', 0)} words")
    print(f"   - Avg answer length: {stats.get('avg_answer_length_words', 0)} words")
    
    # Create data loaders
    print("\n📂 Creating data loaders...")
    train_loader, val_loader = create_agricultural_dataloader(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_split=args.train_split,
        shuffle=True,
    )
    
    if train_loader is None:
        print("❌ Failed to create data loaders")
        return
    
    print(f"✅ Data loaders created:")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    
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
    
    # Train
    print("\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print("-" * 60)
    
    save_path = checkpoint_dir / "best_agri_model.pt"
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=str(save_path),
    )
    
    # Save final model
    final_path = checkpoint_dir / "final_agri_model.pt"
    trainer.save_checkpoint(str(final_path))
    
    # Save tokenizer
    tokenizer_path = checkpoint_dir / "agri_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"📦 Tokenizer saved to {tokenizer_path}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"📦 Models saved to {checkpoint_dir}")
    
    # Training summary
    print("\n📈 Training Summary:")
    print(f"  Total epochs: {len(trainer.train_losses)}")
    print(f"  Final train loss: {trainer.train_losses[-1]:.4f}" if trainer.train_losses else "  No training data")
    print(f"  Final val loss: {trainer.val_losses[-1]:.4f}" if trainer.val_losses else "  No validation data")
    
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AgriNexus AI on agricultural Q&A dataset"
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
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
    
    # Data parameters
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Train/validation split ratio")
    
    # System
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    args = parser.parse_args()
    main(args)
