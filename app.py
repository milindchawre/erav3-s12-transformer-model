import streamlit as st
import torch
import tiktoken
from transformer import GPT, GPTConfig  # Ensure you import your model class

# Load the trained model
@st.cache_resource
def load_model():
    config = GPTConfig()
    model = GPT(config)
    try:
        model.load_state_dict(torch.load('trained_model_quantized.pt'))
        model.eval()  # Set the model to evaluation mode
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return model

# Load the tokenizer
def load_tokenizer():
    return tiktoken.get_encoding('gpt2')

# Generate text function
def generate_text(model, tokenizer, input_text, length, num_sequences):
    # Encode the input text
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    generated_sequences = []
    for _ in range(num_sequences):
        # Generate additional tokens
        with torch.no_grad():
            for _ in range(length):
                logits = model(input_tensor)[0]  # Get logits
                next_token_logits = logits[:, -1, :]  # Get the last token's logits
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)  # Sample from the distribution
                input_tensor = torch.cat((input_tensor, next_token.unsqueeze(0)), dim=1)  # Append the new token

        # Decode the generated tokens
        generated_sequences.append(tokenizer.decode(input_tensor[0].tolist()))

    return generated_sequences

# Streamlit app layout
st.title("GPT Text Generator")
st.write("Enter your text and specify the length of additional text to generate.")

input_text = st.text_area("Input Text", "Once upon a time", max_chars=512)  # Limit to 512 characters
length = st.slider("Predict Additional Text of Length", 1, 50, 10)
num_sequences = st.slider("Number of Sequences to Generate", 1, 5, 1)

if st.button("Generate"):
    model = load_model()
    tokenizer = load_tokenizer()
    st.write("Generating text...")
    generated_texts = generate_text(model, tokenizer, input_text, length, num_sequences)
    st.write("Text generation complete.")

    st.write("Generated Texts:")
    for i, text in enumerate(generated_texts):
        st.subheader(f"Sequence {i + 1}")
        st.write(text)
