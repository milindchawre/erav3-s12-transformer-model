import os
import time
import torch
from transformer import GPT, GPTConfig, DataLoaderLite  # Import your model and data loader

# Initialize the model and data loader
config = GPTConfig()
model = GPT(config)
train_loader = DataLoaderLite(B=4, T=1024)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Function to load the most recent checkpoint
def load_latest_checkpoint(model):
    checkpoint_file = 'checkpoint.pt'
    if not os.path.exists(checkpoint_file):
        return 0  # No checkpoint found, start from epoch 0

    print(f'Loading checkpoint from {checkpoint_file}')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch']

# Load the latest checkpoint if available
start_epoch = load_latest_checkpoint(model)

# Training loop
num_epochs = 91

# Start time tracking
start_time = time.time()

for epoch in range(start_epoch, num_epochs):  # Start from the loaded epoch
    epoch_loss = 0.0  # Initialize epoch loss
    num_steps = 0  # Initialize step counter for the epoch
    last_loss = None  # Variable to store the last loss

    # Calculate total steps for the progress bar
    total_steps = len(train_loader.tokens) // (train_loader.B * train_loader.T)

    # Use tqdm to create a progress bar
    with tqdm(total=total_steps, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
        for step in range(total_steps):  # Iterate over the number of steps
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # Accumulate loss
            num_steps += 1  # Increment step counter
            last_loss = loss.item()  # Store the last loss
            pbar.update(1)  # Update progress bar

            # Check if the loss is below the threshold
            if last_loss < 0.099999:
                print(f'Loss below threshold: {last_loss:.6f}')  # Print loss before breaking
                break  # Exit the loop if the loss condition is met

    # Print the loss at the end of the epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {last_loss:.6f}')

    # Check if the loss condition was met to break out of the epoch loop
    if last_loss < 0.099999:
        print(f'Early stopping at epoch {epoch + 1} due to loss condition met.')
        break  # Exit the epoch loop if the loss condition is met

    # Checkpointing: Save the model and the current epoch after each epoch
    checkpoint_path = 'checkpoint.pt'  # Save to a single checkpoint file
    torch.save({
        'epoch': epoch + 1,  # Save the current epoch number
        'model_state_dict': model.state_dict(),  # Save the model state
    }, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')

# End time tracking
end_time = time.time()
training_duration = end_time - start_time

# Convert training duration to minutes and seconds
minutes = int(training_duration // 60)
seconds = int(training_duration % 60)

# Print the total training time in minute:second format
print(f'Total training time: {minutes} minutes and {seconds} seconds')

# After training your model, apply quantization and save it with compression
def save_model_with_quantization(model, file_path):
    # Switch model to evaluation mode
    model.eval()
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the model to be quantized
        {nn.Linear},  # layers to quantize
        dtype=torch.qint8  # quantization type
    )
    
    # Save the quantized model with compression
    torch.save(quantized_model.state_dict(), file_path, _use_new_zipfile_serialization=True)
    print(f'Model saved to {file_path} with quantization and compression.')

# Call this function after training your model
save_model_with_quantization(model, 'trained_model_quantized.pt')
