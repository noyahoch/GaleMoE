import os
import pickle
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def create_gate_hook(layer_idx):
    """Create a forward hook for a specific gate layer"""
    gate_outputs = []
    
    def hook_fn(module, input, output):
        """Hook function to capture gate outputs"""
        print(f"Gate Layer {layer_idx} - Output shape: {output.shape}")
        print(f"Gate Layer {layer_idx} - Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
        print(f"Gate Layer {layer_idx} - Output argmax indices: {torch.argmax(output, dim=-1)}")
        print(module)
        print()
        
        # Store the output (detach to avoid keeping gradients)
        gate_outputs.append({
            'layer_idx': layer_idx,
            'output': output.detach().cpu().clone(),
            'timestamp': datetime.now().isoformat()
        })
        
        return output
    
    return hook_fn, gate_outputs


def register_gate_hooks(model: torch.nn.Module, num_layers=32):
    """Register forward hooks for all gate modules"""

    gate_module_names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(num_layers)]
    named_modules = {name: module for name, module in model.named_modules()}
    gate_modules = [named_modules.get(gate_name) for gate_name in gate_module_names]

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    all_gate_outputs = []
    for i, module in enumerate(gate_modules):
        hook_fn, gate_outputs = create_gate_hook(i)
        hook_handle = module.register_forward_hook(hook_fn)
        hooks.append(hook_handle)
        all_gate_outputs.append(gate_outputs)
    
    return hooks, all_gate_outputs


def save_gate_outputs(all_gate_outputs, filename=None):
    """Save all gate outputs to a file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gate_outputs_{timestamp}.pkl"
    
    # Flatten the list of lists
    flattened_outputs = []
    for gate_outputs in all_gate_outputs:
        flattened_outputs.extend(gate_outputs)
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(flattened_outputs, f)
    
    print(f"Gate outputs saved to {filepath}")
    print(f"Total captured outputs: {len(flattened_outputs)}")
    
    return filepath


if __name__ == "__main__":
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Register hooks for gate modules
    print("Registering forward hooks for gate modules...")
    hooks, all_gate_outputs = register_gate_hooks(model, num_layers=2)
    
    print(f"Successfully registered {len(hooks)} hooks")

    text = "Hello my name is"
    inputs = tokenizer(text, return_tensors="pt")
    print(f"\nInput tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

    print("\nGenerating text with gate monitoring...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text: {generated_text}")
    
    # Save the captured gate outputs
    output_file = save_gate_outputs(all_gate_outputs)
    
    # Remove hooks to free memory
    for hook in hooks:
        hook.remove()
    
    print("\nHooks removed. Gate monitoring complete.")
