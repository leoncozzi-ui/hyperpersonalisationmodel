import unittest
import torch
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np

from pragma.tokenizer import PragmaTokenizer, SEMANTIC_KEYS, CATEGORICAL_VALUES
from pragma.model import PragmaModel

class TestPragmaPipeline(unittest.TestCase):
    def setUp(self):
        # Initialize tokenizer
        self.tokenizer = PragmaTokenizer()
        
        # Pre-populate some vocabulary words for BPE proxy (descriptions)
        self.tokenizer.add_text_vocabulary([
            "monthly salary payment standard company deposit",
            "british gas electricity utilities bill online payment",
            "emirates airlines ticket purchase flights seat selection",
            "allianz global assistance travel medical insurance premium",
            "customer account manual bank transfer deposit vault"
        ])
        
        # Next Best Action Classes
        self.classes = {
            0: "Savings Option: Open a High-Yield Savings Vault",
            1: "Subscription Upgrade: Offer Premium Plan with Travel Benefits",
            2: "Investment Offer: Purchase Stocks or Crypto ETF",
            3: "No Action / Regular Engagement"
        }
        
        # Build the model (lightweight version)
        self.model = PragmaModel(
            key_vocab_size=len(self.tokenizer.key_to_id),
            value_vocab_size=self.tokenizer.value_vocab_size,
            d_model=32, # 32-dim embeddings for lightweight speed
            n_heads=2,
            num_encoder_layers=1,
            d_ff=64,
            num_classes=len(self.classes)
        )
        
    def create_mock_customers(self):
        evaluation_time = datetime(2026, 5, 26, 12, 0, 0)
        
        # --- Customer A: Regular deposits and bills, then a HUGE deposit ---
        # Intent: High-Yield Savings Vault (Class 0)
        cust_a_profile = {
            "plan": "standard",
            "balance_quantile": "groceries", # balance quantile proxy
            # Lifelong events (key, timestamp)
            "lifelong": [
                ("account_age_milestone", evaluation_time - timedelta(days=365)) # Onboarded 1 year ago
            ]
        }
        cust_a_events = [
            {
                "timestamp": evaluation_time - timedelta(days=3),
                "amount": 2500.0,
                "direction": "in",
                "mcc": "groceries",
                "channel": "in-app",
                "symbol": "eur",
                "description": "monthly salary payment company"
            },
            {
                "timestamp": evaluation_time - timedelta(days=2),
                "amount": 120.0,
                "direction": "out",
                "mcc": "utilities",
                "channel": "email",
                "symbol": "eur",
                "description": "british gas electricity bill"
            },
            {
                "timestamp": evaluation_time - timedelta(minutes=30),
                "amount": 50000.0, # LARGE DEPOSIT
                "direction": "in",
                "mcc": "groceries",
                "channel": "in-app",
                "symbol": "eur",
                "description": "customer account manual bank transfer"
            }
        ]
        
        # --- Customer B: Travel-heavy bookings, then a Large Deposit ---
        # Intent: Upgrade to Premium Plan with Travel insurance / benefits (Class 1)
        cust_b_profile = {
            "plan": "standard",
            "balance_quantile": "dining",
            "lifelong": [
                ("account_age_milestone", evaluation_time - timedelta(days=180)) # Onboarded 6 months ago
            ]
        }
        cust_b_events = [
            {
                "timestamp": evaluation_time - timedelta(days=5),
                "amount": 850.0,
                "direction": "out",
                "mcc": "travel",
                "channel": "in-app",
                "symbol": "usd",
                "description": "emirates airlines ticket purchase flights"
            },
            {
                "timestamp": evaluation_time - timedelta(days=4),
                "amount": 45.0,
                "direction": "out",
                "mcc": "travel",
                "channel": "email",
                "symbol": "usd",
                "description": "allianz global assistance travel insurance"
            },
            {
                "timestamp": evaluation_time - timedelta(minutes=45),
                "amount": 10000.0, # LARGE DEPOSIT
                "direction": "in",
                "mcc": "travel",
                "channel": "in-app",
                "symbol": "usd",
                "description": "customer account manual bank transfer"
            }
        ]
        
        return [(cust_a_profile, cust_a_events, 0), (cust_b_profile, cust_b_events, 1)]
        
    def preprocess_and_pad(self, customers_data) -> Tuple[torch.Tensor, ...]:
        """Preprocesses customers list and pads features into batch tensors."""
        evaluation_time = datetime(2026, 5, 26, 12, 0, 0)
        
        batch_prof_keys = []
        batch_prof_vals = []
        batch_prof_times = []
        
        batch_evt_keys = []
        batch_evt_vals = []
        batch_evt_times = []
        batch_evt_calendar = []
        
        for profile, events, label in customers_data:
            # --- Process Profile ---
            p_keys = []
            p_vals = []
            p_times = []
            
            # Static keys
            for k, v in profile.items():
                if k == "lifelong":
                    continue
                p_keys.append(self.tokenizer.key_to_id[k])
                p_vals.extend(self.tokenizer.tokenize_value(k, v))
                p_times.append(0.0) # Static state has 0 elapsed time
                
            # Lifelong events
            for k, ts in profile.get("lifelong", []):
                p_keys.append(self.tokenizer.key_to_id[k])
                p_vals.extend(self.tokenizer.tokenize_value(k, "standard")) # dummy value
                elapsed = (evaluation_time - ts).total_seconds()
                p_times.append(self.tokenizer.compute_log_seconds(elapsed))
                
            batch_prof_keys.append(p_keys)
            batch_prof_vals.append(p_vals)
            batch_prof_times.append(p_times)
            
            # --- Process Events ---
            e_keys_list = []
            e_vals_list = []
            e_times = []
            e_calendar = []
            
            for evt in events:
                ts = evt["timestamp"]
                elapsed = (evaluation_time - ts).total_seconds()
                e_times.append(self.tokenizer.compute_log_seconds(elapsed))
                
                # Calendar cycles
                h, wd, dm = self.tokenizer.extract_calendar_features(ts)
                e_calendar.append([h, wd, dm])
                
                # Tokens inside event
                keys_tok = []
                vals_tok = []
                for field_key in ["amount", "direction", "mcc", "channel", "symbol", "description"]:
                    val = evt[field_key]
                    field_vals = self.tokenizer.tokenize_value(field_key, val)
                    # Replicate keys for multi-valued fields (e.g. description BPE subwords)
                    keys_tok.extend([self.tokenizer.key_to_id[field_key]] * len(field_vals))
                    vals_tok.extend(field_vals)
                    
                e_keys_list.append(keys_tok)
                e_vals_list.append(evt_vals := vals_tok)
                
            batch_evt_keys.append(e_keys_list)
            batch_evt_vals.append(e_vals_list)
            batch_evt_times.append(e_times)
            batch_evt_calendar.append(e_calendar)
            
        # Padding
        # Profile padding
        max_prof_len = max(len(p) for p in batch_prof_keys)
        pad_key_id = self.tokenizer.key_to_id["[PAD]"]
        pad_val_id = self.tokenizer.cat_to_id["[PAD]"]
        
        padded_prof_keys = []
        padded_prof_vals = []
        padded_prof_times = []
        for pk, pv, pt in zip(batch_prof_keys, batch_prof_vals, batch_prof_times):
            diff = max_prof_len - len(pk)
            padded_prof_keys.append(pk + [pad_key_id] * diff)
            padded_prof_vals.append(pv + [pad_val_id] * diff)
            padded_prof_times.append(pt + [0.0] * diff)
            
        # Event list padding
        max_events = max(len(e) for e in batch_evt_keys)
        max_event_tokens = max(max(len(tokens) for tokens in evts) for evts in batch_evt_keys)
        
        padded_evt_keys = []
        padded_evt_vals = []
        padded_evt_times = []
        padded_evt_calendar = []
        
        for e_keys, e_vals, e_t, e_c in zip(batch_evt_keys, batch_evt_vals, batch_evt_times, batch_evt_calendar):
            # pad individual events
            pe_keys = []
            pe_vals = []
            for ek, ev in zip(e_keys, e_vals):
                diff = max_event_tokens - len(ek)
                pe_keys.append(ek + [pad_key_id] * diff)
                pe_vals.append(ev + [pad_val_id] * diff)
                
            # pad event count
            diff_evt = max_events - len(e_keys)
            for _ in range(diff_evt):
                pe_keys.append([pad_key_id] * max_event_tokens)
                pe_vals.append([pad_val_id] * max_event_tokens)
            padded_evt_keys.append(pe_keys)
            padded_evt_vals.append(pe_vals)
            
            padded_evt_times.append(e_t + [0.0] * diff_evt)
            padded_evt_calendar.append(e_c + [[0, 0, 0]] * diff_evt)
            
        return (
            torch.tensor(padded_prof_keys, dtype=torch.long),
            torch.tensor(padded_prof_vals, dtype=torch.long),
            torch.tensor(padded_prof_times, dtype=torch.float),
            torch.tensor(padded_evt_keys, dtype=torch.long),
            torch.tensor(padded_evt_vals, dtype=torch.long),
            torch.tensor(padded_evt_times, dtype=torch.float),
            torch.tensor(padded_evt_calendar, dtype=torch.float)
        )

    def test_forward_pass(self):
        mock_data = self.create_mock_customers()
        tensors = self.preprocess_and_pad(mock_data)
        
        # Unpack tensors
        p_keys, p_vals, p_times, e_keys, e_vals, e_times, e_cal = tensors
        
        # Model forward call
        with torch.no_grad():
            logits = self.model(
                profile_keys=p_keys,
                profile_values=p_vals,
                profile_times=p_times,
                event_keys=e_keys,
                event_values=e_vals,
                event_times=e_times,
                event_calendar=e_cal
            )
            
        self.assertEqual(logits.shape, (2, 4)) # 2 batch items, 4 output action classes
        print("\n=== SUCCESSFUL FORWARD PASS THROUGH PRAGMA MODEL ===")
        print(f"Logits Shape: {logits.shape}")
        print(f"Customer A Logits: {logits[0].tolist()}")
        print(f"Customer B Logits: {logits[1].tolist()}")
        
    def test_mock_training_step(self):
        mock_data = self.create_mock_customers()
        p_keys, p_vals, p_times, e_keys, e_vals, e_times, e_cal = self.preprocess_and_pad(mock_data)
        labels = torch.tensor([data[2] for data in mock_data], dtype=torch.long) # [0, 1]
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Run a 20-step mock gradient descent to verify gradient calculation
        self.model.train()
        print("\n=== RUNNING MOCK GRADIENT DESCENT TRAINING STEPS ===")
        for epoch in range(20):
            optimizer.zero_grad()
            logits = self.model(
                profile_keys=p_keys,
                profile_values=p_vals,
                profile_times=p_times,
                event_keys=e_keys,
                event_values=e_vals,
                event_times=e_times,
                event_calendar=e_cal
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            print(f"Step {epoch+1}/20 - Classification Loss: {loss.item():.4f}")

            
        # Evaluate trained outputs
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                profile_keys=p_keys,
                profile_values=p_vals,
                profile_times=p_times,
                event_keys=e_keys,
                event_values=e_vals,
                event_times=e_times,
                event_calendar=e_cal
            )
            predictions = torch.argmax(logits, dim=-1)
            
        print(f"Predictions: {predictions.tolist()} (Expected: {labels.tolist()})")
        self.assertEqual(predictions.tolist(), labels.tolist())
        print("Predictions matched the expected intents perfectly!")
        print("Customer A next best action: ", self.classes[predictions[0].item()])
        print("Customer B next best action: ", self.classes[predictions[1].item()])

if __name__ == "__main__":
    unittest.main()
