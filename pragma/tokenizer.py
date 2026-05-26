import math
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Predefined lists of keys and categorical options for tokenisation
SEMANTIC_KEYS = [
    "[PAD]", "[UNK]", "[MASK]", "[USR]", "[EVT]",
    "amount", "direction", "mcc", "channel", "symbol", 
    "plan", "balance_quantile", "account_age_milestone",
    "description"
]

CATEGORICAL_VALUES = [
    "[PAD]", "[UNK]", "[MASK]",
    # direction
    "in", "out",
    # channel
    "email", "push", "sms", "in-app",
    # plan
    "standard", "plus", "premium", "metal", "ultra",
    # mcc (Merchant Category Codes - sample)
    "groceries", "dining", "travel", "utilities", "investment", "crypto",
    # symbol (trading)
    "AAPL", "GOOGL", "BTC", "ETH", "EUR", "USD"
]

class PragmaTokenizer:
    def __init__(self, amount_percentiles: List[float] = None):
        # Key mapping
        self.key_to_id = {key: idx for idx, key in enumerate(SEMANTIC_KEYS)}
        self.id_to_key = {idx: key for key, idx in self.key_to_id.items()}
        
        # Categorical value mapping
        self.cat_to_id = {val: idx for idx, val in enumerate(CATEGORICAL_VALUES)}
        self.id_to_cat = {idx: val for val, idx in self.cat_to_id.items()}
        
        # Numerical binning for amount.
        # If none, use standard log-percentiles
        if amount_percentiles is None:
            self.amount_bins = [0.0, 5.0, 15.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, 10000.0]
        else:
            self.amount_bins = amount_percentiles
            
        # Word vocabulary for free text descriptions (BPE proxy)
        self.word_to_id = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}
        
    def add_text_vocabulary(self, texts: List[str]):
        """Populates vocab for text fields (BPE proxy)."""
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = len(self.word_to_id)
                    
    def tokenize_value(self, key: str, value: Any) -> List[int]:
        """Maps a value to a list of token IDs depending on the key type."""
        if key == "amount":
            # Percentile binning
            val = float(value)
            if val == 0:
                # Reserve bin 0 for zero values
                bin_idx = 0
            else:
                bin_idx = 1
                for limit in self.amount_bins:
                    if val > limit:
                        bin_idx += 1
                    else:
                        break
            # Offset by a safe range (e.g., categoricals length + 1000) to keep vocabs separated,
            # or we can use a single unified vocabulary mapping. Let's use unified values vocabulary:
            # Categoricals take indices: 0 to len(cat_to_id)-1
            # Amount bins take indices: len(cat_to_id) to len(cat_to_id) + len(bins)
            return [len(self.cat_to_id) + bin_idx]
            
        elif key in ["direction", "channel", "plan", "mcc", "symbol"]:
            # Categorical value
            val_str = str(value).lower()
            val_id = self.cat_to_id.get(val_str, self.cat_to_id["[UNK]"])
            return [val_id]
            
        elif key == "description":
            # Text value: Tokenize BPE style (word-level proxy)
            val_str = str(value).lower()
            tokens = []
            for word in val_str.split():
                tokens.append(self.word_to_id.get(word, self.word_to_id["[UNK]"]))
            return tokens if tokens else [self.word_to_id["[PAD]"]]
            
        else:
            # Generic categorical treatment
            val_str = str(value).lower()
            return [self.cat_to_id.get(val_str, self.cat_to_id["[UNK]"])]
            
    def compute_log_seconds(self, seconds_elapsed: float) -> float:
        """Applies soft logarithmic scaling to elapsed time.
        Formula from paper: 8 * ln(1 + t / 8)
        """
        return 8.0 * math.log(1.0 + max(0.0, seconds_elapsed) / 8.0)
        
    def extract_calendar_features(self, dt: datetime) -> Tuple[int, int, int]:
        """Decomposes timestamp into hour-of-day, day-of-week, day-of-month."""
        return dt.hour, dt.weekday(), dt.day

    @property
    def value_vocab_size(self) -> int:
        # Unified value vocabulary size = categoricals + numerical bins + textual words
        return len(self.cat_to_id) + len(self.amount_bins) + 2 + len(self.word_to_id)
