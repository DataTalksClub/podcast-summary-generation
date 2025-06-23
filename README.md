# 📢 Podcast Summaries Project

## 🚀 Overview
This project aims to generate structured **PDF reports** from podcast interviews, highlighting key takeaways, quotes, and insights. The goal is to create **shareable** and **accessible** summaries for a broader audience.

## 🔹 Features (Planned)
- **Summarization using LLMs**
- **Search & Retrieval**
- **PDF Report Generation**
- **Web UI (Streamlit)** for user interaction

## 📌 Getting Started
### **1️⃣ Setup the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/DataTalksClub/podcast-summary-generation.git
   cd podcast-summaries
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Before running the application, ensure you have your `OPEN_API_KEY` and `GROK_API_KEY` configured.

You can do this using **either** of the following methods:

#### Option 1: Use a `.env` File

Create a `.env` file in the root of your project with the following content:

```env
OPEN_API_KEY = sample-value-here
GROK_API_KEY = sample-value-here
```

#### Option 2: Use a `secrets.toml` File

Create a file named `secrets.toml` inside the `.streamlit` folder with the following content:

```bash
OPEN_API_KEY = sample-value-here
GROK_API_KEY = sample-value-here
```

### **2️⃣ Run the Project (Development Mode)**

To run the application, do the following:

```bash
# Start the backend services (if needed)
docker-compose up -d

# Generate a podcast summary using the OpenAI API key.
 python main_openai.py  --input episode.md --output episode_summary_openai.md

# Run the Streamlit application
python run_streamlit_app.py
```

## 🔄 Workflow Pipeline
1. **LLM Processing** → Summarization, Extracting Key Insights
2. **Storage & Retrieval** → Search Engine (ElasticSearch/In-memory DB)
3. **PDF Generation** → Formatted Report
4. **Web UI** → User Interaction & Downloads

## 🏗️ Contribution Guidelines
- Open an issue before working on any feature.
- Use feature branches for development.
- Submit PRs with at least **2 approvals** before merging.

## 📚 Resources
- [Project Documentation](docs/README.md)
- [GitHub Issues](https://github.com/DataTalksClub/podcast-summary-generation/issues)

🚀 **Let's build something great together!** 🎙️📄
