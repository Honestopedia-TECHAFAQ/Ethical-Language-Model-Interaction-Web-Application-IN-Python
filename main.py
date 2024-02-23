import streamlit as st
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pytesseract

model_path = "path_to_your_fine_tuned_model.pt"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_path)
def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text
def main():
    st.title("Ethical Language Model Interaction")
    user_query = st.text_area("Input your query:")
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if st.button("Generate Response"):
        if user_query:
            with st.spinner("Generating..."):
                if image_file is not None:
                    image = Image.open(image_file)
                    extracted_text = extract_text_from_image(image)
                    prompt = user_query + " " + extracted_text
                else:
                    prompt = user_query
                response = generate_response(prompt)
            st.success("Generated Successfully!")
            st.write("Model Response:")
            st.write(response)
        else:
            st.warning("Please input a query.")

if __name__ == "__main__":
    main()
