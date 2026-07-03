# AI Career Assistant

A Streamlit application that uses the Hugging Face Inference API to provide career advice, interview questions, placement guidance, and resume analysis.

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in this folder:
   ```bash
   echo "HF_TOKEN=your_huggingface_token" > .env
   ```

4. Run the app:
   ```bash
   streamlit run ai_agent.py
   ```

## Notes

- The app requires a valid Hugging Face API token stored in `.env` as `HF_TOKEN`.
- Do not commit your `.env` file to GitHub.
