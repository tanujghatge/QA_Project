# Playlist-Powered Chatbot

## Overview

This project is a Playlist-Powered Chatbot that uses a combination of OpenAI's technologies and other tools to create a conversational AI system. It takes a playlist URL as input, extracts transcriptions from videos in the playlist, and builds a chatbot that can answer user queries based on the content in the playlist.

## Features

- **Transcription Extraction**: Utilizes the Whisper JAX library to extract transcriptions from videos linked in the provided playlist URL.

- **Semantic Search**: Uses OpenAI's embeddings to convert user queries and transcriptions into vectors and then performs semantic search to retrieve the most relevant transcriptions for generating responses.

- **Question-Answering Retriever Chain**: Implements the Langchain Question-Answering retriever chain to extract specific answers from the retrieved transcriptions.

- **Efficient Vector Database**: Employs Faiss as a vector database for efficient storage and retrieval of transcript embeddings.

- **OpenAI Chat GPT**: Utilizes OpenAI's Chat GPT API for generating natural language responses based on user queries and relevant transcriptions.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Necessary Python packages (see `requirements.txt`)

## Getting Started

1. Clone this repository:

   ```shell
   git clone https://github.com/tanujghatge/QA_Project.git
   cd QA_Project
   ```

2. Install the required Python packages:

   ```shell
   pip install -r requirements.txt
   ```


3. Run the application:

   ```shell
   python app.py
   ```

4. Access the chatbot through the provided user interface or API endpoints (customize based on your implementation).

## Usage

- To interact with the chatbot, simply input a user query. The chatbot will respond with relevant answers from the transcriptions.

- Customize the UI or API endpoints to suit your application's requirements.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI](https://openai.com) for providing the Chat GPT API and embeddings technology.
- [Langchain](https://langchain.io) for the Question-Answering retriever chain.
- [Facebook AI Research](https://github.com/facebookresearch/faiss) for the Faiss vector database.

## Contributors

- Tanuj Ghatge (https://github.com/tanujghatge)
## Contact

For questions or inquiries about the project, please contact ghatgetanuj@gmail.com.

---

Feel free to customize this README file further to include specific installation instructions, usage examples, and any other relevant information about your project. Make sure to replace placeholders like 
