# 📧 Cold Email Generator

An AI-powered cold email generator that extracts job requirements from URLs and creates personalized cold emails based on your portfolio and skills.

## 🚀 Features

- Extract job postings from company websites
- AI-powered job requirement analysis
- Automatic skill matching with your portfolio
- Personalized cold email generation
- Portfolio link recommendations

## 🛠️ Setup Instructions

### For Streamlit Cloud Deployment

1. **Clone this repository** to your GitHub account

2. **Get a GROQ API Key** (Free):
   - Visit [https://console.groq.com/](https://console.groq.com/)
   - Sign up for a free account
   - Navigate to API Keys section
   - Create a new API key

3. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Deploy this repository

4. **Configure Secrets**:
   - In your Streamlit Cloud app dashboard
   - Go to Settings → Secrets
   - Add the following:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```

### For Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anup2003D/Cold-Email-Generator.git
   cd Cold-Email-Generator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ API key
   ```

4. **Run the application**:
   ```bash
   streamlit run CMGmain.py
   ```

## 📁 Portfolio Configuration

Update your portfolio information in `Resource/my_portfolio.csv`:

```csv
Techstack,Links
"Python, Machine Learning, Data Science","https://github.com/yourusername/ml-project"
"React, JavaScript, Frontend Development","https://github.com/yourusername/react-app"
```

## 🔧 Technical Details

### Architecture

- **CMGmain.py**: Main Streamlit application
- **CMGchain.py**: LangChain integration with GROQ LLM
- **CMGportfolio.py**: Portfolio management with ChromaDB
- **CMGutils.py**: Utility functions for text processing

### Key Features

- **Lazy Loading**: ChromaDB is only initialized when needed
- **Error Handling**: Graceful handling of missing API keys
- **Session State**: Efficient state management in Streamlit
- **Responsive Design**: Works on desktop and mobile

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY environment variable is not set"**
   - Make sure you've added your API key to Streamlit secrets or .env file

2. **ChromaDB initialization errors**
   - The app now uses lazy loading, so ChromaDB only initializes when needed

3. **Import errors**
   - Run `pip install -r requirements.txt` to install all dependencies

### Getting Help

- Check the error messages in the Streamlit interface
- Verify your GROQ API key is valid
- Ensure your portfolio CSV file exists and has the correct format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [GROQ](https://groq.com/) LLM
- Uses [LangChain](https://langchain.com/) for AI orchestration
- Vector storage with [ChromaDB](https://www.trychroma.com/)
