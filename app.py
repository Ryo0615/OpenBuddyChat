import os
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Hugging FaceのAPIトークンを設定
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    st.write("HUGGINGFACE_TOKEN not found!")
else:
    os.environ["HUGGINGFACE_TOKEN"] = token

model_name_or_path = "TheBloke/OpenBuddy-Llama2-13B-v11.1-GGUF"
model_basename = "openbuddy-llama2-13b-v11.1.Q2_K.gguf"

try:
    model_path = hf_hub_download(
        repo_id=model_name_or_path, filename=model_basename, revision="main"
    )
    st.write(f"Model downloaded to {model_path}")
except Exception as e:
    st.write(f"Error downloading model: {e}")

try:
    llama = Llama(model_path)
except Exception as e:
    st.write(f"Error initializing Llama: {e}")

def predict(messages):
    try:
        # Llamaでの回答を取得（ストリーミングオン）
        streamer = llama.create_chat_completion(messages, stream=True)

        partial_message = ""
        for msg in streamer:
            message = msg["choices"][0]["delta"]
            if "content" in message:
                partial_message += message["content"]
                yield partial_message
    except Exception as e:
        st.write(f"Error in prediction: {e}")

def main():
    st.title("Chat with ChatGPT Clone!")

    # Session state for retaining messages
    if not hasattr(st.session_state, 'messages'):
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}")

    # Input for the user message
    user_message = st.chat_input("Your Message")

    # React to user input
    if user_message:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(f"{user_message}")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for char in predict(
                [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            ):
                full_response = char  # += char
                message_placeholder.markdown(full_response + " ❚ ")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

if __name__ == "__main__":
    main()
