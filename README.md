# Reddit User Persona Generator

A Python script that scrapes Reddit user profiles and generates detailed user personas using LangChain v0.3 and the Groq API. This enhanced version includes additional sections for social/professional status, experience tier, archetype, motivations, behavior habits, and MBTI-style personality percentages.

---

## Features

- **Scrapes Reddit Profiles**: Collects posts and comments from user profiles.
- **Data Modeling**: Uses dataclasses and Pydantic models to structure scraped data and persona outputs.
- **LangChain v0.3 Integration**: Leverages the latest LangChain pipelines for prompt templates and chaining.
- **Groq API for LLM Processing**: Generates personas using Groq LLM (e.g., llama-3.3-70b-versatile).
- **Enhanced Persona Sections**:
  - Demographics: name, age range, location, occupation
  - Social/Professional Status and Skill Tier
  - Archetype (e.g., Explorer, Creator, Helper)
  - Interests, Goals, Frustrations
  - Motivations with intensity scores (0–100)
  - Behavior & Habits (posting frequency, interaction style)
  - MBTI-Style Personality Percentages (I/E, N/S, F/T, P/J)
  - Communication style, technology comfort, and social media behavior
  - Citations for each characteristic from actual posts/comments
- **Output Writer**: Formats and writes a comprehensive persona report to a text file.

---

## Requirements

- Python 3.8+
- [LangChain v0.3](https://github.com/langchain-ai/langchain)
- `langchain-groq` client library
- `requests`, `beautifulsoup4`, `pydantic`, `python-dotenv`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/qusaii21/reddit-persona.git
   cd reddit-persona
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\\Scripts\\activate   # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

1. Create a `.env` file in the project root:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   ```
2. Ensure your Reddit profile URLs follow the format `https://www.reddit.com/user/<username>/`.

---

## Usage

Run the script with a single profile URL:

```bash
python main.py --url https://www.reddit.com/user/username/ 
```

Or without arguments to use default example profiles:

```bash
python persona_builder.py
```

Example output:

```
Processing user: kojied
--------------------------------------------------
Found 48 posts/comments
✓ Enhanced persona generated and saved to: output/kojied_persona.txt
```

---

## Output

- Persona files are written to the `output` directory, named `<username>_persona.txt`.
- Detailed logging is available in `reddit_scraper.log`.

---


## Logging

Logs are written to `reddit_scraper.log` and stdout. Logging level is set to INFO by default.

---


