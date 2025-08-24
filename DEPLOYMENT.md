# 🚀 Alpha Co-Pilot Deployment Guide - Vercel + Docker

Complete guide to deploy your Alpha Co-Pilot Flask application to Vercel using Docker containerization.

## 📋 Prerequisites

- [Docker](https://www.docker.com/get-started) installed locally
- [Vercel CLI](https://vercel.com/cli) installed (`npm i -g vercel`)
- [Git](https://git-scm.com/) repository with your code
- API keys for Azure OpenAI (required) and optionally Gemini

## 🔧 Environment Setup

### 1. Create Environment File

Create a `.env` file in your project root with these required variables:

```bash
# REQUIRED: Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=o4-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# REQUIRED: Flask Settings
FLASK_SECRET_KEY=your_super_secret_flask_key_change_this_in_production
FLASK_ENV=production

# OPTIONAL: Enhanced Features
GEMINI_API_KEY=your_gemini_api_key_here
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url
CMC_API_KEY=your_coinmarketcap_pro_api_key
```

### 2. API Keys Setup Guide

#### Azure OpenAI (Required)
1. Go to [Azure Portal](https://portal.azure.com)
2. Create an Azure OpenAI resource
3. Deploy the `o4-mini` model
4. Copy the API key and endpoint from the resource overview

#### Google Gemini (Optional - Enhanced Analysis)
1. Visit [Google AI Studio](https://ai.google.dev)
2. Create a new API key
3. Enable Google Search grounding for real-time data

#### Discord Webhook (Optional - Social Sharing)
1. Open your Discord server
2. Go to Server Settings → Integrations → Webhooks
3. Create New Webhook and copy the URL

## 🐳 Local Docker Testing

### Build and Test Locally

```bash
# Build the Docker image
docker build -t alpha-co-pilot .

# Run locally with environment variables
docker run -p 5000:5000 --env-file .env alpha-co-pilot

# Or use Docker Compose for easier development
docker-compose up --build
```

### Test the Application

```bash
# Health check
curl http://localhost:5000/health

# Test API endpoint
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "BTC analysis"}'
```

## 🌐 Vercel Deployment

### Method 1: Using Vercel CLI (Recommended)

```bash
# 1. Login to Vercel
vercel login

# 2. Initialize project (in your project directory)
vercel

# 3. Set environment variables
vercel env add AZURE_OPENAI_API_KEY
vercel env add AZURE_OPENAI_ENDPOINT
vercel env add AZURE_OPENAI_DEPLOYMENT_NAME
vercel env add AZURE_OPENAI_API_VERSION
vercel env add FLASK_SECRET_KEY
vercel env add GEMINI_API_KEY
vercel env add DISCORD_WEBHOOK_URL

# 4. Deploy
vercel --prod
```

### Method 2: Using Vercel Dashboard

1. **Connect Repository**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your Git repository

2. **Configure Build Settings**
   - Framework Preset: `Other`
   - Build Command: `docker build -t alpha-co-pilot .`
   - Output Directory: Leave empty (Docker handles this)

3. **Set Environment Variables**
   - Go to Project Settings → Environment Variables
   - Add all required variables from your `.env` file
   - Make sure to set them for Production, Preview, and Development

4. **Deploy**
   - Click "Deploy"
   - Vercel will automatically build and deploy your Docker container

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Docker Build Fails
```bash
# Check Docker is running
docker version

# Build with verbose output
docker build -t alpha-co-pilot . --progress=plain

# Check for file permission issues
chmod +x run.py
```

#### 2. Environment Variables Not Loading
```bash
# Verify environment variables in Vercel
vercel env ls

# Test locally with env file
docker run -p 5000:5000 --env-file .env alpha-co-pilot
```

#### 3. API Timeout Issues
- Increase timeout in `vercel.json` (already set to 30s)
- Check Azure OpenAI model deployment status
- Verify API keys are correct and have proper permissions

#### 4. Memory Issues
- Current Docker image uses Python 3.11-slim for efficiency
- Vercel function memory is set to 1024MB in `vercel.json`
- Consider upgrading Vercel plan for more resources if needed

### Health Check Endpoints

```bash
# Basic health check
curl https://your-app.vercel.app/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-01-08T10:30:00Z",
  "langchain_llm": "available",
  "gemini": "available",
  "model_type": "reasoning_model"
}
```

## 🚀 Production Optimization

### 1. Performance Tuning

**Dockerfile Optimizations:**
- Multi-stage builds for smaller images
- Non-root user for security
- Health checks for monitoring
- Gunicorn with 2 workers for production

**Application Optimizations:**
- Token usage monitoring and limits
- API rate limiting and caching
- Error handling and fallback mechanisms

### 2. Monitoring and Logging

```bash
# View Vercel function logs
vercel logs

# Check specific deployment
vercel logs --since=1h
```

### 3. Environment-Specific Configurations

```bash
# Development
FLASK_ENV=development
FLASK_DEBUG=True

# Production (Vercel automatically sets these)
FLASK_ENV=production
FLASK_DEBUG=False
PORT=5000
```

## 🔐 Security Best Practices

1. **Environment Variables**
   - Never commit `.env` files to version control
   - Use Vercel's secure environment variable storage
   - Rotate API keys regularly

2. **Docker Security**
   - Non-root user in container
   - Minimal base image (Python slim)
   - No sensitive data in Docker layers

3. **Application Security**
   - CORS properly configured
   - Input validation on all endpoints
   - Rate limiting on API endpoints

## 📊 Scaling Considerations

### Vercel Limits
- **Function timeout**: 30 seconds (configurable)
- **Memory**: 1024MB (configurable up to 3GB on Pro)
- **Concurrent executions**: 1000 (Pro plan)

### Optimization Tips
- Use Redis for caching (add to docker-compose.yml)
- Implement request queuing for high load
- Consider migrating to dedicated hosting for very high traffic

## 🎯 Quick Deployment Checklist

- [ ] Docker builds successfully locally
- [ ] All required environment variables set
- [ ] Health endpoint returns 200
- [ ] API endpoints tested with real data
- [ ] Vercel CLI authenticated
- [ ] Environment variables added to Vercel
- [ ] Deployment successful and accessible
- [ ] Discord webhook tested (if configured)
- [ ] Azure OpenAI quota sufficient for traffic

## 📱 Alternative Deployment Options

If Vercel doesn't meet your needs, this Docker setup also works with:

- **Railway**: `railway up`
- **Render**: Connect Git repo with Docker
- **DigitalOcean App Platform**: Docker-based deployment
- **Google Cloud Run**: `gcloud run deploy`
- **AWS ECS**: Container-based deployment

## 🆘 Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Verify all environment variables are correctly set
3. Test Docker build locally first
4. Check Vercel function logs for error details
5. Ensure Azure OpenAI deployment is active and has quota

---

**🎉 Your Alpha Co-Pilot should now be live on Vercel!**

Access your deployed app at: `https://your-project-name.vercel.app`
