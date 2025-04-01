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

### **2️⃣ Run the Project (Development Mode)**
```bash
# Start the backend services (if needed)
docker-compose up -d

# Run the application
python main.py
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
