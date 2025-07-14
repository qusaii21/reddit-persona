# Reddit User Persona Builder

This tool takes a Reddit user profile URL, scrapes their posts and comments, and uses a LLaMA model (via the Groq API) to generate a **User Persona**, along with inline “citations” back to the scraped content.

## Features
1. Accepts a Reddit profile URL (e.g. `https://www.reddit.com/user/kojied/`).
2. Scrapes the user’s latest posts & comments (via PRAW / Reddit API).
3. Builds a Persona document with thumbnail sections (e.g. **Interests**, **Tone & Style**, **Top Topics**, etc.).
4. For each persona characteristic, includes the post/comment URL(s) that informed it.
5. Outputs to a text file named `<reddit_username>_persona.txt`.
6. **Sample outputs** for `kojied` and `Hungry-Move-6603` are in `samples/`.

## Prerequisites
- Python 3.8+
- A Reddit API app (client ID/secret)  
- A Groq API key (for LLaMA inference)

## Installation

```bash
git clone https://github.com/yourorg/reddit-persona.git
cd reddit-persona
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
