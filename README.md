# 🧞 ShopGenie-E: Intelligent Multi-Agent Shopping Assistant

**ShopGenie-E** is a next-generation e-commerce assistant built with Streamlit and Python. Unlike standard search engines, it utilizes a Multi-Agent System (MAS) to understand user intent, retrieve real-time data, reason through specifications, and provide bias-free product rankings with visual dashboards.

![ShopGenie Banner](shopgenie_logo.png)

## ✨ Key Features

* **🧞 Multi-Agent Backend:**
    * **Intent Agent:** Distinguishes between casual chat, budget constraints, and specific product queries.
    * **Retrieval Agent:** Scours the web/database for real-time product availability.
    * **Reasoner Agent:** Analyzes specs against user needs to assign a "Genie Score" (0-10).
    * **Image Fetcher:** Retrieves high-quality product imagery dynamically.
* **📊 Dynamic Dashboard:** Automatically renders comparison tables and product cards when a search is complete.
* **🎨 Custom UI/UX:** A "Genie Purple & Gold" theme with custom CSS for buttons, cards, and metrics.
* **📝 Deep Dive Mode:** Click on any product to see translated benefits, technical specs, and dealbreakers.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python)
* **Data Manipulation:** Pandas, JSON
* **Backend Logic:** Custom `backend.py` (LLM orchestration)
* **Styling:** Custom CSS Injection

## 📂 Project Structure

```text
ShopGenie-E/
├── app.py                 # Main Streamlit Application (Frontend)
├── backend.py             # Agent Logic (Intent, Retrieval, Reasoner)
├── shopgenie_logo.png     # Project Logo (Required for UI)
├── requirements.txt       # Python Dependencies
└── README.md              # Project Documentation
