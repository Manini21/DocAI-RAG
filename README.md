# DocAI RAG 📚🤖

### Because searching through 50 pages of a PDF just to find one answer isn't exactly fun.

Hey! Welcome to DocAI RAG.

This is a project I built while exploring **Retrieval-Augmented Generation (RAG)** and trying to understand how AI applications can work with information that isn't already part of an LLM's knowledge.

The idea was simple: What if I could upload a PDF, ask it questions in plain English, and get answers based on what's actually written in the document?

So, I decided to build it.

## 💡 What does it do?

DocAI RAG lets you upload PDF documents and have a conversation with their contents.

You can:
- Upload and process PDF documents.
- Ask questions about the uploaded content.
- Retrieve relevant passages instead of feeding the entire document to the LLM.
- Get AI-generated answers grounded in the retrieved information.
- See source references, including the document name and page number.

It also supports questions that go beyond the document. When appropriate, the assistant can combine document context with general knowledge and distinguish between the two.

## 🔍 How I built it

I wanted to understand what happens behind the scenes, so I broke the application into smaller pieces.

Here's the journey a PDF takes:

1. **PDF loading:** PyMuPDF extracts text from the uploaded PDF.
2. **Chunking:** The extracted text is split into smaller chunks using Recursive Character Text Splitter (512 characters with 64 characters of overlap).
3. **Embeddings:** Sentence Transformers converts those chunks into numerical representations using `all-MiniLM-L6-v2`.
4. **Storage:** ChromaDB stores the embeddings along with the original text and metadata.
5. **Retrieval:** When a question comes in, the application retrieves the most relevant chunks using vector similarity.
6. **Generation:** Groq's Llama 3.3 70B model uses the retrieved context to generate an answer.

The overall idea is to retrieve first and generate afterwards. That's the RAG part!

## 🛠️ Tech Stack

| Technology | What I used it for |
|---|---|
| Python | Core application logic |
| Streamlit | User interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Local text embeddings |
| ChromaDB | Vector database |
| Groq | LLM inference |
| OpenAI GPT-OSS 120B (via Groq) | Answer generation |
| LangChain Text Splitters | Text chunking |

## 🧠 Things I learned along the way

This project taught me more than just how to connect an LLM to a PDF.

A few things that stood out:

- **Embeddings aren't just fancy keywords.** I learned how text can be represented as vectors so that semantically similar content can be retrieved.
- **Chunking matters.** Splitting documents isn't as straightforward as cutting text into equal pieces. Overlap helps preserve context between chunks.
- **RAG isn't the same as fine-tuning.** I learned that I could give an LLM access to external information without training the model itself.
- **Vector databases have a purpose.** I got hands-on experience storing embeddings, querying them, and retrieving relevant document chunks.
- **LLMs can sound confident and still be wrong.** This made me pay more attention to prompts, retrieved context, and source references.
- **Debugging is part of building.** I also explored alternatives when my initial embedding API approach ran into quota issues, eventually moving to local Sentence Transformers.

One of my biggest takeaways: getting an answer from an LLM is easy. Getting an answer that's actually grounded in the right information is the interesting part.

## ⚙️ Run it locally

Want to try it? Here's how to get started.

**1. Clone the repository**

```bash
git clone https://github.com/Manini21/DocAI-RAG.git
cd DocAI-RAG
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up your environment variables**

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

**5. Run the application**

```bash
streamlit run app.py
```

The app should open in your browser.

## 🚧 What's next?

This project is a learning journey, not something I consider finished.

Some things I'd like to explore next:
- Improving retrieval quality.
- Experimenting with hybrid search and reranking.
- Adding support for more document formats.
- Evaluating answer quality and retrieval accuracy.
- Exploring more advanced RAG techniques.

## 👩‍💻 A little note

I built this project to move beyond simply reading about RAG and actually understand how its pieces fit together.

There were things I didn't understand initially, things that didn't work on the first attempt, and plenty of small discoveries along the way.

If you're also learning about RAG, I hope this project gives you a useful starting point.

And if you have suggestions or ideas for improving it, I'd love to hear them!

---

**Built with curiosity, Python, and a lot of debugging. ☕**
