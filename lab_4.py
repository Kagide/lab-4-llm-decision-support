#!/usr/bin/env python
# coding: utf-8

# # Lab 4: LLMs and Prompt Engineering for Decision Support
# 
# **Duration:** 2 weeks [30 Jul - 13 Aug, 2026]
# **Due Date:** 13th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
# 
# **Student Name:** Kagiraneza Egide
# **Student ID:** 29082028
# 
# ---
# 
# ### Objective
# 
# In the previous labs you *trained* models. In this lab you will *use* a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
# 
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
# 
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
# 
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
# 
# ---
# 
# ### Choosing an API provider
# 
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
# 
# | Provider | Free tier | Notes |
# |---|---|---|
# | **Groq** (recommended) | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini** | Yes | `google-generativeai` package |
# | **Hugging Face Inference API** | Yes, limited | Many open models |
# | OpenAI / Anthropic | Paid | Fine if you already have credits |
# 
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.

# ---
# ### Part 0: Repository and API-key setup
# 
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
# 
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.

# In[1]:


import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from the Lab 4 folder
env_path = r"C:\Users\kagid\lab-4-llm\.env"
load_dotenv(env_path, override=True)

API_KEY = os.getenv("GROQ_API_KEY")

print("API key loaded:", bool(API_KEY))

if not API_KEY:
    raise ValueError("GROQ_API_KEY was not found in the .env file.")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

print("Client configured for Groq.")


# ---
# # Section 1 — Talking to an LLM Programmatically
# 
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).

# ### Part 1.1 — Your first API call

# In[2]:


# If this cell returns 401, check that your .env contains your newly created Groq key.
# reusable helper function for the whole lab.
def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
            temperature=0.7, max_tokens=500, show_usage=False):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if show_usage:
        print("Token usage:", response.usage)

    return response.choices[0].message.content

# first simple API call.
answer = ask_llm(
    "In one sentence, what is the main purpose of a microfinance loan?",
    temperature=0.2,
    show_usage=True
)
print(answer)


# **Student Reasoning — Anatomy of a call**
# 
# *1. What is the difference between the `system` and `user` roles? Give an example of something that belongs in each.*
# 
# > **Answer:** The `system` message sets the model's general rules for the task, while the `user` message gives the actual request. In this lab, the system message can tell the model to stay factual and not invent missing information, while the user message contains the loan letter and asks for a summary or extraction.
# 
# *2. What is a token, roughly? Why do API providers bill per token rather than per request?*
# 
# > **Answer:** A token is a small unit of text that the model reads or generates. Since LLMs generate text token by token, a long application and a long answer use more model work than a short request. That is why token-based billing reflects usage better than charging the same amount for every request.
# 

# ### Part 1.2 — Temperature: the randomness dial

# In[3]:


#same question, repeated at two temperatures.
test_question = "Suggest a name for a savings product for market traders in Accra."

print("TEMPERATURE = 0.0")
for i in range(5):
    result = ask_llm(test_question, temperature=0.0, max_tokens=80)
    print(f"{i + 1}. {result}")

print("\nTEMPERATURE = 1.2")
for i in range(5):
    result = ask_llm(test_question, temperature=1.2, max_tokens=80)
    print(f"{i + 1}. {result}")


# **Student Reasoning — Temperature**
# 
# > **Answer:** In my runs, temperature 0.0 produced very similar answers, while temperature 1.2 gave more variation in the suggested names and wording. This matches the idea from the notes that lower temperature gives more weight to high-probability tokens, while higher temperature allows more variation. For extraction and factual summaries in this lab, I would keep the temperature low because consistency matters more than creativity.
# 

# ---
# # Section 2 — The Dataset: Loan Application Letters
# 
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
# 
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.

# In[4]:


LETTERS = {
"L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",

"L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",

"L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",

"L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",

"L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",

"L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
  "L001": {"applicant_name": "Akosua Mensah", "amount_ghs": 8000,  "purpose": "buy deep freezer / expand into frozen foods",
           "monthly_profit_ghs": 900,  "has_collateral_or_guarantor": True,  "repayment_months": 20},
  "L003": {"applicant_name": "Efua Darko",    "amount_ghs": 15000, "purpose": "industrial sewing machines and fabric stock",
           "monthly_profit_ghs": 2800, "has_collateral_or_guarantor": True,  "repayment_months": 15},
  "L006": {"applicant_name": "Kofi",          "amount_ghs": 50000, "purpose": "car wash, provision shop, phone imports",
           "monthly_profit_ghs": None, "has_collateral_or_guarantor": False, "repayment_months": 12},
}

print(f"{len(LETTERS)} letters loaded.")


# ---
# # Section 3 — Prompt Engineering for the Decision Support System
# 
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.

# ### Part 3.1 — Component 1: Summarization
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.

# In[5]:


# ADDED: V1 is intentionally simple so I can compare it with a better prompt.
SUMMARY_PROMPT_V1 = "Summarize this loan application:\n\n{}"

# ADDED: V2 gives the model a role and clear limits.
SUMMARY_SYSTEM_V2 = """You are an assistant to a microfinance loan officer.
Summarize loan applications in a factual and neutral way.
Use only information stated in the letter. Do not invent or assume missing details.
Write only 3-4 sentences and focus on the applicant, amount, purpose, repayment ability,
and any collateral, guarantor, savings, or important risk mentioned."""

SUMMARY_PROMPT_V2 = "Summarize this loan application:\n\n{}"

summary_results = {}

for letter_id in ["L002", "L006"]:
    letter = LETTERS[letter_id]

    v1 = ask_llm(SUMMARY_PROMPT_V1.format(letter), temperature=0.7)
    v2 = ask_llm(
        SUMMARY_PROMPT_V2.format(letter),
        system_prompt=SUMMARY_SYSTEM_V2,
        temperature=0.0
    )

    summary_results[letter_id] = {"V1": v1, "V2": v2}

    print(f"\n===== {letter_id} =====")
    print("\nV1:")
    print(v1)
    print("\nV2:")
    print(v2)


# **Student Reasoning — Summarization prompts**
# 
# *1. What concrete problems did V1's output have that V2 fixed? Quote examples.*
# 
# > **Answer:** V1 was only told to "Summarize this loan application," so the model decided for itself what style and details to include. For L006, V1 used a longer list, while V2 gave a shorter factual paragraph and clearly stated that the businesses had not started and no collateral was provided. For L002, V2 also kept the important facts together: the GHS 25,000 request, trotro repair and personal debts, slow business, no collateral, and unclear repayment plan. The main improvement was better control of length, tone, and grounding.
# 
# *2. Why is "no invented details" an essential instruction in this application? What is this failure mode called in the LLM literature?*
# 
# > **Answer:** A loan officer may use the output as part of a real review, so an invented income, collateral item, or repayment detail could mislead the decision. The notes describe unsupported generated information as **hallucination**. In this system, the model should stay with what the applicant actually stated and mark missing information instead of filling gaps.
# 

# ### Part 3.2 — Component 2: Structured extraction (JSON)
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.

# In[6]:


import json
import pandas as pd

#one made-up example, separate from the six lab letters.
FEW_SHOT_LETTER = """Dear Manager,
My name is Ama Kusi. I run a small bakery and I am requesting GHS 6,000 to buy a new oven.
My monthly profit is about GHS 1,200. I can repay over 10 months. My mother will be my guarantor.
"""

FEW_SHOT_JSON = {
    "applicant_name": "Ama Kusi",
    "amount_ghs": 6000,
    "purpose": "buy a new oven",
    "monthly_profit_ghs": 1200,
    "has_collateral_or_guarantor": True,
    "repayment_months": 10
}

# ADDED: explicit extraction instructions and exact schema.
EXTRACT_SYSTEM_PROMPT = f"""You extract structured facts from microfinance loan application letters.
Return ONLY one valid JSON object. Do not add markdown or explanation.
Use only information directly stated in the letter. If a field is not stated, use null. Do not guess.

Return EXACTLY these keys:
{{
  "applicant_name": string or null,
  "amount_ghs": number or null,
  "purpose": string or null,
  "monthly_profit_ghs": number or null,
  "has_collateral_or_guarantor": boolean,
  "repayment_months": number or null
}}

For has_collateral_or_guarantor, return true only if the letter states collateral or a guarantor.

Worked example:
LETTER:
{FEW_SHOT_LETTER}

JSON:
{json.dumps(FEW_SHOT_JSON)}
"""

EXTRACT_PROMPT = "Extract the required fields from this letter:\n\n{}"

#temperature is a parameter because Part 4.2 tests two values.
def extract_fields(letter_text, temperature=0.0):
    raw = ask_llm(
        EXTRACT_PROMPT.format(letter_text),
        system_prompt=EXTRACT_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=300
    )

    # Remove common markdown fences if the model adds them anyway.
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        print("Warning: model output was not valid JSON.")
        print("Raw output:", raw)
        return None

#run extraction for all six letters and put the results in a DataFrame.
extracted_results = {}

for letter_id, letter_text in LETTERS.items():
    extracted_results[letter_id] = extract_fields(letter_text)

extracted_df = pd.DataFrame.from_dict(extracted_results, orient="index")
extracted_df.index.name = "letter_id"

display(extracted_df)


# **Student Reasoning — Structured extraction**
# 
# *1. Why must the few-shot example NOT come from the six letters you are processing?*
# 
# > **Answer:** If I use one of the six target letters as the worked example, I would be showing the model one of the cases that I later evaluate. A separate example teaches the format without leaking part of the evaluation set.
# 
# *2. Why "use null, do not guess" — what did the model do without that instruction?*
# 
# > **Answer:** Some letters do not contain every required field. Without a clear instruction, the model may fill a gap with a plausible value. Using `null` makes it clear that the information was not stated, which is safer than turning a guess into a stored fact.
# 
# *3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?*
# 
# > **Answer:** Extraction is supposed to return the same stated facts in the same structure, so extra variation is not useful. A low temperature makes high-probability outputs more dominant, which fits this task. In a creative task such as naming a product, more variation can be useful.
# 

# ### Part 3.3 — Component 3: The decision-support brief
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.

# In[7]:


#final decision-support prompt.
BRIEF_SYSTEM_PROMPT = """You are an assistant supporting a human microfinance loan officer.
Your job is to organize evidence, not to make the lending decision.
Use only the application letter and extracted data provided. Do not invent facts.
Do not say APPROVE or REJECT. Final lending decisions must be made by a human officer.

Return these four sections:
1. Strengths
2. Risks / red flags
3. Missing information to request
4. Suggested next step

Use short bullet points under the first three sections.
The suggested next step may recommend an interview, document request, verification,
or senior review, but it must not make the final lending decision."""

BRIEF_PROMPT = """Application letter:
{letter}

Extracted data:
{extracted}

Prepare the decision-support brief."""

def make_brief(letter_text, extracted):
    return ask_llm(
        BRIEF_PROMPT.format(
            letter=letter_text,
            extracted=json.dumps(extracted, indent=2)
        ),
        system_prompt=BRIEF_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=500
    )

#generate all six briefs.
briefs = {}
for letter_id, letter_text in LETTERS.items():
    briefs[letter_id] = make_brief(letter_text, extracted_results[letter_id])

for letter_id in ["L003", "L006"]:
    print(f"\n===== DECISION-SUPPORT BRIEF: {letter_id} =====")
    print(briefs[letter_id])


# **Student Reasoning — Decision support**
# 
# *1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the system identify the right strengths and red flags in each?*
# 
# > **Answer:** Yes. L003 has an existing registered business, GHS 2,800 average monthly profit, sales records, a fixed deposit that can be pledged, and a specific repayment plan. L006 is much less supported because the three businesses have not started, there is no current profit stated, there is no collateral, and repayment depends on the businesses becoming successful. The brief is useful when it organizes these facts and missing information without making the final decision.
# 
# *2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and one ethical reason.*
# 
# > **Answer:** Practically, one letter does not contain everything a lender may need to verify before a final decision. Ethically, the notes emphasize bias, accountability, and keeping a human in the loop. The model should support the loan officer, not replace the officer's judgment.
# 

# ### Part 3.4 — Commit your prompt templates
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into
# a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
# 
# > **Commit hash:** `a24bc54`
# 
# **Commit used:** `Complete Lab 4 LLM project`
# 

# ---
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
# 
# An impressive demo is not a trustworthy system. Now measure it.

# ### Part 4.1 — Extraction accuracy against gold labels

# In[8]:


#compare extracted results with the gold labels field by field.
fields = list(next(iter(GOLD.values())).keys())
evaluation_rows = []

for field in fields:
    row = {"field": field}
    correct_count = 0

    for letter_id in GOLD:
        predicted = extracted_results[letter_id].get(field) if extracted_results[letter_id] else None
        expected = GOLD[letter_id][field]

        if field == "applicant_name" and predicted is not None and expected is not None:
            is_correct = str(predicted).strip().lower() == str(expected).strip().lower()
        else:
            is_correct = predicted == expected

        row[letter_id] = "Correct" if is_correct else f"Wrong ({predicted!r})"
        correct_count += int(is_correct)

    row["accuracy"] = correct_count / len(GOLD)
    evaluation_rows.append(row)

accuracy_df = pd.DataFrame(evaluation_rows).set_index("field")
display(accuracy_df)

print("Overall field-level accuracy:",
      f"{sum(r['accuracy'] for r in evaluation_rows) / len(evaluation_rows):.2%}")


# ### Part 4.2 — Reliability: is the system consistent?

# In[9]:


from collections import Counter

#reliability experiment for L004.
def reliability_test(temperature):
    runs = [extract_fields(LETTERS["L004"], temperature=temperature) for _ in range(5)]

    valid_runs = [r for r in runs if r is not None]
    normalized = [json.dumps(r, sort_keys=True) for r in valid_runs]
    unique_outputs = len(set(normalized))

    print(f"\nTemperature = {temperature}")
    print(f"Valid JSON: {len(valid_runs)}/5")

    if valid_runs:
        most_common_count = Counter(normalized).most_common(1)[0][1]
        print(f"Largest group of identical outputs: {most_common_count}/5")
        print(f"Unique valid outputs: {unique_outputs}")
    else:
        print("Largest group of identical outputs: 0/5")
        print("Unique valid outputs: 0")

    return runs

runs_temp_0 = reliability_test(0.0)
runs_temp_1 = reliability_test(1.0)


# ### Part 4.3 — Hallucination probing

# In[10]:


#two adversarial tests for hallucination.
# Test 1: ask for information that L001 never gives.
test1_prompt = f"""Read the loan application below and answer one question.
Use only the letter. If the information is not stated, say exactly: NOT STATED.

Question: What is the applicant's credit score?

Letter:
{LETTERS['L001']}
"""

test1_output = ask_llm(
    test1_prompt,
    system_prompt="You answer only from the supplied loan application and never invent missing facts.",
    temperature=0.0,
    max_tokens=80
)

test1_pass = "not stated" in test1_output.lower()
print("TEST 1 OUTPUT (verbatim):")
print(test1_output)
print("RESULT:", "PASS" if test1_pass else "FAIL")

# Test 2: irrelevant text should not turn into a fake applicant.
irrelevant_text = """Accra will be partly cloudy today with light winds.
Temperatures are expected to rise during the afternoon before becoming cooler in the evening."""

test2_output = extract_fields(irrelevant_text, temperature=0.0)
print("\nTEST 2 OUTPUT (verbatim):")
print(json.dumps(test2_output, indent=2))

expected_null_fields = [
    "applicant_name", "amount_ghs", "purpose",
    "monthly_profit_ghs", "repayment_months"
]
nulls_ok = test2_output is not None and all(test2_output.get(k) is None for k in expected_null_fields)
boolean_ok = test2_output is not None and test2_output.get("has_collateral_or_guarantor") is False
print("RESULT:", "PASS" if nulls_ok and boolean_ok else "FAIL")


# **Student Reasoning — Evaluation results**
# 
# *1. Report your extraction accuracy. Which field was hardest for the model and why?*
# 
# > **Answer:** My field-level exact-match accuracy was **83.33%**. The hardest field was `purpose`, which scored 0% in the exact comparison. Looking at the outputs, the extracted purposes were mostly paraphrases of the gold labels rather than different meanings. This shows a limitation of exact-string evaluation for free-text fields, while the numeric and boolean fields were easier to compare directly.
# 
# *2. What did the reliability experiment show about temperature and production systems?*
# 
# > **Answer:** In this experiment, both temperature 0.0 and 1.0 produced valid JSON in all five runs, and all five outputs were identical at each temperature. So the higher temperature did not reduce consistency in this small test. I would still use a low temperature for production extraction because the task needs stable facts, not creative variation.
# 
# *3. Did your system hallucinate under probing? If yes, how could the prompt (or the system design around it) reduce the risk?*
# 
# > **Answer:** It did not hallucinate in these two probes. For the missing credit score it returned `NOT STATED`, and for unrelated weather text it returned null values instead of inventing an applicant. I would still keep the grounding instruction, validate the JSON, and send uncertain cases to human review because two passing tests do not prove that hallucination cannot happen.
# 

# ### Part 4.4 — Appropriateness: should this system exist?
# No code in this part — just judgment, which is the scarcest skill in AI for business.

# **Student Reasoning — Appropriateness**
# 
# *1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions with your system, who could be unfairly harmed, and how? Consider applicants who write poorly in English but run solid businesses.*
# 
# > **Answer:** Applicants with weaker English or less experience writing formal business documents could be treated unfairly even when their businesses are strong. The model may respond to how well the letter is written instead of only to the financial evidence. This connects to the bias and fairness concern in the notes, so writing style should not become an accidental credit-scoring feature.
# 
# *2. Loan letters contain personal data. What are the implications of sending them to a third-party API in another country? What would you check before deploying this at a real Ghanaian microfinance institution?*
# 
# > **Answer:** The letters contain names, financial information, and business details, so sending them to an external API creates privacy and data-governance concerns. The notes also treat privacy as an important AI deployment issue. Before deployment, I would check the provider's data retention and training policy, where data is processed, access controls, deletion rules, and the institution's data-protection requirements. I would also send only the information the model actually needs.
# 
# *3. Name TWO concrete safeguards you would build around this system in production (think: human review points, logging, appeal processes, monitoring).*
# 
# > **Answer:** First, a human loan officer should review the original application and supporting documents before any final lending decision. Second, I would keep an audit log of the model input, extracted facts, recommendation, model version, and final human action so mistakes can be traced and reviewed.
# 

# ---
# # Section 5 — Reflection
# 
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from iterating on the model hyperparameters you tuned in Lab 3?
# 
# > **Answer:** Both involve changing something, testing the result, and comparing whether the output improves. In Lab 3 I changed settings that affected how my own model learned. In this lab I am not retraining the LLM; I am changing the instructions, examples, and generation settings given to an already trained model. In both cases, testing matters because small changes can affect the final output.
# 
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended? What single evaluation result most influenced your answer?
# 
# > **Answer:** No. I would use it as decision support, not as an unattended decision maker. The result that influenced me most was the **83.33% exact-match extraction score** because it shows that even a structured task needs checking. The hallucination probes passed, which is encouraging, but they were only two tests. This fits the human-in-the-loop idea from the notes: AI should assist human judgment rather than make the final call by itself.
# 
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to process 1,000 applications per month. What does that imply for provider choice?
# 
# > **Answer:** My first API test used **104 total tokens**, so 1,000 calls of that same size would use about **104,000 tokens**. A real application in this lab needs longer prompts and several calls for summarization, extraction, and the brief, so the true monthly total would be higher. I would therefore compare providers on token price, rate limits, reliability, and privacy instead of using the first-call estimate as the final production cost.
# 
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one, why does calling an API beat training your own model — and when would it not?
# 
# > **Answer:** For this task, an API is practical because the model already handles natural language, so I can focus on prompts and evaluation instead of collecting a very large text dataset and training a large model. I would consider a local or custom model when privacy is critical, the task is very specialized, I have enough suitable data, API cost becomes too high, or I need more control over deployment.
# 

# ---
# ### Submission checklist
# 
# - [ ] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [ ] **No API key anywhere in the notebook or the commit history.**
# - [ ] Every **Student Reasoning** box is filled in with full sentences.
# - [ ] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [ ] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [ ] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [ ] Repository link submitted to the course portal.
# - [ ] AI Declaration form in Repository.

# In[ ]:




