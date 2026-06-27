"""
Data Loading and Preprocessing

Handles loading agricultural datasets and preparing them for training.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader


class AgriculturalDataset(Dataset):
    """Custom dataset for agricultural Q&A pairs."""
    
    def __init__(
        self,
        qa_pairs: List[Dict],
        tokenizer,
        max_length: int = 128,
    ):
        """
        Initialize dataset.
        
        Args:
            qa_pairs: List of Q&A dictionaries
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self):
        """Prepare examples by combining Q&A."""
        examples = []
        
        for qa in self.qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            
            # Combine question and answer
            combined_text = f"Q: {question} A: {answer}"
            
            examples.append({
                "text": combined_text,
                "question": question,
                "answer": answer,
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["text"]
        
        # Tokenize with truncation and padding
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad to max_length
            pad_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * pad_length
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def load_qa_dataset(filepath: str) -> List[Dict]:
    """Load Q&A dataset from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get("qa_pairs", [])


def create_agricultural_dataloader(
    dataset_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    train_split: float = 0.8,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from agricultural dataset.
    
    Args:
        dataset_path: Path to agricultural QA dataset
        tokenizer: Tokenizer for encoding
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        train_split: Train/val split ratio
        shuffle: Whether to shuffle data
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Load dataset
    if not Path(dataset_path).exists():
        print(f"Warning: Dataset not found at {dataset_path}")
        return None, None
    
    qa_pairs = load_qa_dataset(dataset_path)
    
    if not qa_pairs:
        print("Warning: No QA pairs found in dataset")
        return None, None
    
    # Create dataset
    dataset = AgriculturalDataset(
        qa_pairs=qa_pairs,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    # Split into train and validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader


def create_knowledge_base_from_dataset(dataset_path: str) -> List[str]:
    """
    Extract facts from Q&A dataset for knowledge base.
    
    Args:
        dataset_path: Path to agricultural QA dataset
    
    Returns:
        List of facts extracted from answers
    """
    if not Path(dataset_path).exists():
        return []
    
    qa_pairs = load_qa_dataset(dataset_path)
    
    facts = []
    for qa in qa_pairs:
        answer = qa.get("answer", "")
        if answer:
            facts.append(answer)
    
    return facts


def get_dataset_stats(dataset_path: str) -> Dict:
    """Get statistics about the dataset."""
    if not Path(dataset_path).exists():
        return {}
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    qa_pairs = data.get("qa_pairs", [])
    
    total_questions = len(qa_pairs)
    avg_question_length = sum(
        len(qa.get("question", "").split())
        for qa in qa_pairs
    ) / max(1, total_questions)
    avg_answer_length = sum(
        len(qa.get("answer", "").split())
        for qa in qa_pairs
    ) / max(1, total_questions)
    
    return {
        "total_qa_pairs": total_questions,
        "avg_question_length_words": round(avg_question_length, 2),
        "avg_answer_length_words": round(avg_answer_length, 2),
        "metadata": data.get("metadata", {}),
    }


if __name__ == "__main__":
    # Example usage
    from src.tokenizer import CharTokenizer
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.vocab = {
        "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
        **{chr(i): i + 4 for i in range(32, 127)}
    }
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    
    # Load and display dataset stats
    dataset_path = "data/agricultural_qa_dataset.json"
    stats = get_dataset_stats(dataset_path)
    print("Dataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Create dataloaders
    train_loader, val_loader = create_agricultural_dataloader(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=8,
        max_length=128,
    )
    
    if train_loader:
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Show sample batch
        sample_batch = next(iter(train_loader))
        print(f"\nSample batch shape: {sample_batch['input_ids'].shape}")
