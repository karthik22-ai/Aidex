# Aidex
# 🤖 Aidex: Your Medical assistant

Aidex is an AI-powered virtual assistant designed to act as a brain that listens, thinks, and responds like a human—combining voice, vision, and reasoning into a seamless interface. Built with FastAPI and a modern frontend, Aidex leverages LLMs and real-time input to deliver personalized, human-centric interactions.

---

## 🚀 Features

- 🌐 Frontend (HTML, JS): Interactive UI for chat with AI brain.
- ⚙️ Backend (FastAPI): Handles requests, integrates with AI agents.
- 🧠 LLM/Agents: Powered by Gemini/OpenAI with multi-agent logic.
- 🎤 Voice Support (Pluggable): Expandable architecture for voice integration.
- 📡 Deployable on Render with GitHub CI/CD.

---

## 📁 Folder Structure

```bash
Aidex/
├── backend/
│   ├── main.py          # FastAPI backend entry point
│   ├── agents.py        # Custom AI agent logic
│   ├── requirements.txt # Python dependencies
├── frontend/
│   ├── index.html       # Frontend interface
├── render.yaml          # Deployment config for Render.com
├── README.md            # You’re here!
