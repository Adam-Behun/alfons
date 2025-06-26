# Alfons: Prior Authorization Bot

Alfons is a real-world prior authorization bot for healthcare, enabling phone-based conversations, real-time database updates, and a web interface to view live logs.

## Features
- **Telephony**: Make/receive calls via Twilio with a testable phone number.
- **Voice Interaction**: ElevenLabs for human-like TTS/STT.
- **Conversational AI**: xAI Grok 3 via LangChain for natural, empathetic responses.
- **Database Updates**: Real-time logging and data extraction to Supabase.
- **Web Interface**: Next.js with real-time updates via Supabase subscriptions, hosted on Vercel.
- **Escalation**: Forwards complex queries to a human phone number.
- **HIPAA Compliance**: Uses mock data for demo safety.
- **GitHub Workflow**: CI/CD with GitHub Actions for Vercel deployment.

## Prerequisites
- Windows 10/11
- VSCode with Python and Continue extensions
- Node.js (v18+)
- Python (v3.10+)
- Git
- Twilio, ElevenLabs, Supabase, and xAI accounts

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/alfons.git
   cd alfons