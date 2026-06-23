"""
Character-level Tokenizer

Simple tokenizer that maps characters to integer tokens.
Can be extended to support subword tokenization (BPE, WordPiece).
"""

import json
from typing import List, Dict, Tuple


class CharTokenizer:
    """Character-level tokenizer for text encoding/decoding."""
    
    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Dictionary mapping characters to token IDs.
                   If None, will be built from training data.
        """
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()} if vocab else {}
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Reserve special tokens if starting fresh
        if not self.vocab:
            self.vocab = {
                "<PAD>": self.pad_token_id,
                "<UNK>": self.unk_token_id,
                "<BOS>": self.bos_token_id,
                "<EOS>": self.eos_token_id,
            }
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from.
        """
        # Start with special tokens
        vocab = {
            "<PAD>": self.pad_token_id,
            "<UNK>": self.unk_token_id,
            "<BOS>": self.bos_token_id,
            "<EOS>": self.eos_token_id,
        }
        
        # Add unique characters
        idx = 4
        for text in texts:
            for char in set(text):
                if char not in vocab:
                    vocab[char] = idx
                    idx += 1
        
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode.
            add_special_tokens: Whether to add BOS and EOS tokens.
        
        Returns:
            List of token IDs.
        """
        tokens = []
        
        # Add BOS token
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        # Encode characters
        for char in text:
            token_id = self.vocab.get(char, self.unk_token_id)
            tokens.append(token_id)
        
        # Add EOS token
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in output.
        
        Returns:
            Decoded text.
        """
        text = ""
        special_token_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            
            char = self.inv_vocab.get(token_id, "?")
            if char not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                text += char
        
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def save(self, filepath: str):
        """Save tokenizer vocabulary to file."""
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def load(self, filepath: str):
        """Load tokenizer vocabulary from file."""
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
            self.inv_vocab = {int(v): k for k, v in self.vocab.items()}
