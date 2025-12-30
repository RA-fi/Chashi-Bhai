# চাষী ভাই (Chashi Bhai) - AI-Powered Agricultural Assistant

An intelligent agricultural assistant powered by AI, providing farmers with real-time weather data, crop recommendations, pest management advice, and agricultural insights in both Bengali and English.

## 🚀 Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

### Prerequisites
- A [Railway.app](https://railway.app) account
- A GitHub account
- An OpenAI or Groq API key (for AI features)

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/chashi-bhai.git
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to [Railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically detect the configuration

3. **Set Environment Variables**
   In Railway's dashboard, add these environment variables:
   
   **Required:**
   - `OPENAI_API_KEY` or `GROQ_API_KEY` - Your AI API key
   
   **Optional:**
   - `NASA_API_KEY` - Get from [NASA API](https://api.nasa.gov/) (defaults to DEMO_KEY)
   - `ALLOW_ORIGINS` - Your frontend URL (e.g., `https://yourusername.github.io`)

4. **Deploy**
   - Railway will automatically deploy your application
   - You'll get a public URL like: `https://your-app.railway.app`

## 🛠️ Local Development

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chashi-bhai.git
   cd chashi-bhai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the application**
   ```bash
   uvicorn backend:app --host 0.0.0.0 --port 8080 --reload
   ```

6. **Open in browser**
   ```
   http://localhost:8080
   ```

## 📋 Features

- 🌾 **Crop Recommendations** - AI-powered crop suggestions based on location and season
- 🌡️ **Weather Information** - Real-time weather data from NASA and other sources
- 🐛 **Pest Management** - Identify and manage agricultural pests
- 📊 **Agricultural Insights** - Data-driven farming advice
- 🌍 **Bilingual Support** - Available in Bengali (বাংলা) and English
- 🛰️ **Satellite Data** - NASA satellite imagery and agricultural data
- 💬 **AI Chat Assistant** - Interactive agricultural advisory

## 🔧 Technology Stack

- **Backend:** FastAPI (Python)
- **AI/ML:** LangChain, OpenAI/Groq
- **Data Sources:** NASA POWER, MODIS, DuckDuckGo, Wikipedia, ArXiv
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Railway.app

## 📝 Environment Variables

See [.env.example](.env.example) for all available configuration options.

### Required Variables
- `OPENAI_API_KEY` or `GROQ_API_KEY` - AI model API key

### Optional Variables
- `NASA_API_KEY` - NASA API access (defaults to DEMO_KEY)
- `NASA_EARTHDATA_TOKEN` - Enhanced satellite data access
- `WEATHER_UNDERGROUND_API_KEY` - Premium weather data
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8080, Railway sets this automatically)
- `ALLOW_ORIGINS` - CORS allowed origins

## 🚀 API Endpoints

- `GET /` - Main application interface
- `POST /chat` - AI chat endpoint
- `GET /weather/{location}` - Weather data
- `GET /crop-recommendation` - Crop recommendations
- `GET /health` - Health check endpoint

## 📦 Project Structure

```
chashi-bhai/
├── backend.py              # FastAPI application
├── settings.py            # Configuration settings
├── requirements.txt       # Python dependencies
├── Procfile              # Railway deployment config
├── railway.json          # Railway settings
├── .env.example          # Environment template
├── index.html            # Frontend
└── assets/
    ├── css/
    ├── js/
    ├── audio/
    └── svg/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- NASA for satellite and weather data APIs
- OpenAI/Groq for AI models
- LangChain for AI orchestration
- FastAPI for the web framework

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ for farmers
