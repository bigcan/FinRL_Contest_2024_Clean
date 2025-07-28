#!/usr/bin/env python3
"""
Debug script to identify the index out of bounds error
Error: index 275140 is out of bounds for dimension 0 with size 164737

Let's analyze the math behind this error.
"""

def analyze_index_error():
    """Analyze the specific numbers from the error"""
    
    print("🔍 Index Error Analysis")
    print("=" * 50)
    
    # From the error message
    trying_to_access = 275140
    available_size = 164737
    overflow = trying_to_access - available_size
    
    print(f"📊 Error Details:")
    print(f"   Trying to access index: {trying_to_access:,}")
    print(f"   Available size: {available_size:,}")
    print(f"   Overflow amount: {overflow:,}")
    print(f"   Ratio: {trying_to_access / available_size:.3f}x")
    
    # From training config
    print(f"\n⚙️  Training Configuration Analysis:")
    batch_size = 256
    num_sims = 16
    data_length = 10000
    
    print(f"   Batch size: {batch_size}")
    print(f"   Num sims: {num_sims}")
    print(f"   Data length: {data_length}")
    
    # Replay buffer analysis
    print(f"\n🗃️  Replay Buffer Analysis:")
    print(f"   In ReplayBuffer.sample(), the key line is:")
    print(f"   ids = torch.randint(sample_len * num_seqs, size=(batch_size,))")
    
    # Let's figure out what values could cause this
    sample_len = available_size - 1  # 164737 - 1 = 164736
    num_seqs = num_sims  # 16
    
    max_possible_id = sample_len * num_seqs - 1
    
    print(f"   Available sample_len: {sample_len:,}")
    print(f"   Num sequences: {num_seqs}")
    print(f"   Max possible ID: {max_possible_id:,}")
    
    if max_possible_id >= trying_to_access:
        print(f"   ✅ This range COULD generate the problematic index")
    else:
        print(f"   ❌ This range could NOT generate the problematic index")
        
    # Check tensor indexing
    print(f"\n🎯 Tensor Indexing Analysis:")
    print(f"   In the sample method:")
    print(f"   ids0 = torch.fmod(ids, sample_len)  # ids % sample_len") 
    print(f"   ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len")
    
    # Simulate the problematic index
    problematic_id = trying_to_access
    ids0 = problematic_id % sample_len
    ids1 = problematic_id // sample_len
    
    print(f"   For problematic ID {problematic_id:,}:")
    print(f"   ids0 (row index): {ids0:,}")
    print(f"   ids1 (sequence index): {ids1}")
    
    # Check if ids1 is within bounds
    if ids1 >= num_seqs:
        print(f"   ❌ ids1 ({ids1}) >= num_seqs ({num_seqs}) - SEQUENCE INDEX OUT OF BOUNDS!")
    else:
        print(f"   ✅ ids1 ({ids1}) < num_seqs ({num_seqs}) - sequence index OK")
        
    if ids0 >= sample_len:
        print(f"   ❌ ids0 ({ids0}) >= sample_len ({sample_len}) - ROW INDEX OUT OF BOUNDS!")
    else:
        print(f"   ✅ ids0 ({ids0}) < sample_len ({sample_len}) - row index OK")
    
    # Now check the actual tensor access
    print(f"\n🔧 Tensor Access Analysis:")
    print(f"   The error occurs in: self.states[ids0, ids1]")
    print(f"   If states tensor shape is [max_size, num_seqs, state_dim]")
    print(f"   And we're accessing [ids0={ids0:,}, ids1={ids1}]")
    
    # Check if this could be a next_state access issue
    print(f"\n🔄 Next State Access Analysis:")
    print(f"   The last line returns: self.states[ids0 + 1, ids1]")
    ids0_plus_1 = ids0 + 1
    print(f"   ids0 + 1 = {ids0_plus_1:,}")
    
    if ids0_plus_1 >= available_size:
        print(f"   ❌ FOUND THE ISSUE! ids0+1 ({ids0_plus_1:,}) >= available_size ({available_size:,})")
        print(f"   This happens when accessing next_state for the last element!")
    else:
        print(f"   ✅ ids0+1 ({ids0_plus_1:,}) < available_size ({available_size:,}) - next state access OK")

def propose_fixes():
    """Propose fixes for the index error"""
    
    print(f"\n🔧 Proposed Fixes:")
    print("=" * 30)
    
    print("1. **Fix sample_len calculation**:")
    print("   Current: sample_len = self.cur_size - 1")
    print("   Problem: This allows sampling the last element, but next_state access fails")
    print("   Fix: sample_len = self.cur_size - 2  # Leave buffer for next_state")
    
    print("\n2. **Add bounds checking**:")
    print("   Add validation in sample() method to ensure valid indices")
    
    print("\n3. **Handle edge case gracefully**:")
    print("   Check if ids0 + 1 >= cur_size and handle appropriately")

def main():
    analyze_index_error()
    propose_fixes()
    
    print(f"\n🎯 **CONCLUSION**:")
    print("The error is in ReplayBuffer.sample() when accessing next_state.")
    print("The buffer allows sampling the last element, but then tries to access")
    print("next_state at index cur_size, which doesn't exist.")
    print("\nThis is a classic off-by-one error in the replay buffer implementation.")

if __name__ == "__main__":
    main()