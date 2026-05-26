import sys
import os
from datetime import datetime, timedelta

# Import model and tokenizer components
from pragma.tokenizer import PragmaTokenizer
from pragma.model import PragmaModel
from tests.test_model import TestPragmaPipeline

def main():
    print("==================================================================")
    print("       PRAGMA-BASED HYPER-PERSONALISATION MODEL DEMO              ")
    print("==================================================================")
    print("This script demonstrates next-best-action calculation based on ")
    print("multi-source event history and static customer profile states.")
    print("Inspired by Revolut's PRAGMA paper (arXiv:2604.08649).\n")
    
    # 1. Setup Pipeline & Mock Data
    print("[1/4] Initializing components and preparing model...")
    pipeline = TestPragmaPipeline()
    pipeline.setUp()
    
    mock_data = pipeline.create_mock_customers()
    print(f"  -> Loaded {len(mock_data)} customer scenario templates.")
    
    # 2. Pad and preprocess batch data into PyTorch tensors
    print("[2/4] Preprocessing and tokenizing multi-source customer sequences...")
    tensors = pipeline.preprocess_and_pad(mock_data)
    p_keys, p_vals, p_times, e_keys, e_vals, e_times, e_cal = tensors
    print(f"  -> Padded Profile Keys Shape: {p_keys.shape}")
    print(f"  -> Padded Event Keys Shape:   {e_keys.shape}")
    
    # 3. Training the model on mock customer profiles to learn customer intent
    print("[3/4] Fine-tuning PRAGMA parameters on downstream intent labels...")
    pipeline.test_mock_training_step()
    pipeline.model.eval()

    import torch
    with torch.no_grad():
        logits = pipeline.model(
            profile_keys=p_keys,
            profile_values=p_vals,
            profile_times=p_times,
            event_keys=e_keys,
            event_values=e_vals,
            event_times=e_times,
            event_calendar=e_cal
        )
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
    print("\n[4/4] Evaluation Results:")
    for idx, (profile, events, _) in enumerate(mock_data):
        cust_name = f"Customer {'A' if idx == 0 else 'B'}"
        pred_class = predictions[idx].item()
        pred_prob = probabilities[idx][pred_class].item() * 100
        
        print(f"\n------------------------------------------------------------------")
        print(f"📊 {cust_name} Portfolio & History:")
        print(f"  • Plan Type:           {profile['plan'].upper()}")
        print(f"  • Balance Quantile:    {profile['balance_quantile'].upper()}")
        print(f"  • Account Milestones:  Onboarded {(idx == 0 and '1 year' or '6 months')} ago")
        print(f"  • Event History count: {len(events)} events")
        
        print("  • Last 3 Events:")
        for e in events:
            time_diff = datetime(2026, 5, 26, 12, 0, 0) - e['timestamp']
            hours = int(time_diff.total_seconds() // 3600)
            minutes = int((time_diff.total_seconds() % 3600) // 60)
            time_str = f"{hours}h {minutes}m ago" if hours > 0 else f"{minutes}m ago"
            
            symbol_sign = "+" if e['direction'] == "in" else "-"
            print(f"    - [{time_str}] {e['mcc'].upper()} event ({e['direction']}): {symbol_sign}${e['amount']:.2f} ({e['description']})")
            
        print(f"\n🔮 Next-Best-Action Decision:")
        print(f"  🚀 Recommended Action: \033[1;32m{pipeline.classes[pred_class]}\033[0m")
        print(f"  🎯 Model Confidence:   {pred_prob:.2f}%")
        
    print("\n------------------------------------------------------------------")
    print("💡 Real-World Pre-training and Downstream Adaptations:")
    print("In production, PRAGMA is pre-trained via self-supervised Masked ")
    print("Language Modeling (MLM) on billions of transaction records first.")
    print("For each specific downstream task (like Credit Scoring, Product ")
    print("Recommendations, or Uplift Models), the backbone is adapted with ")
    print("Parameter-Efficient Fine-Tuning (LoRA) which yields massive gains")
    print("while reusing 98% of the pre-trained parameters!")
    print("==================================================================\n")

if __name__ == "__main__":
    main()
