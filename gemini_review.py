import google.generativeai as genai
import os
import re
from pathlib import Path

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

diff_dir = Path("diffs")
review_content = ["## Code Review"]

for diff_file in diff_dir.glob("*.diff"):
    with open(diff_file, "r") as f:
        diff = f.read()
    
    # Extract filename from diff header
    filename_match = re.search(r'^diff --git a/(.+?) b/', diff, re.MULTILINE)
    filename = filename_match.group(1) if filename_match else diff_file.stem

    response = model.generate_content(
        f"""Analyze this code diff as a senior developer. Focus on:
1. Potential bugs and errors
2. Security vulnerabilities
3. Code quality improvements
4. Best practices violations

Format response in markdown with clear sections. Be concise but thorough.

Code diff:
{diff}"""
    )
    
    review_content.append(f"### 📄 File: {filename}\n\n{response.text}\n\n---")

with open("review.md", "w") as f:
    f.write("\n".join(review_content))
