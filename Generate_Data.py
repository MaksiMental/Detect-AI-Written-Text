import pandas as pd
import openai
import time

# OpenAI API key
openai.api_key = "INSERT API KEY HERE"

# Load the dataset
df = pd.read_csv("./data/gpt/persuade.csv")


# Function to generate an essay based on instructions
def generate_essay(instruction):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Please write the essay as one continuous text block without using bullet points.",
                },
                {"role": "user", "content": instruction},
            ],
            temperature=0.8,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["?"],
        )
        if response.choices and len(response.choices) > 0:
            essay = response.choices[0].message.content.strip()
            return essay
        else:
            print("No choices found in the response.")
            return None
    except Exception as e:
        print(f"Error generating essay: {e}")
        return None


# Initialize list to store results
essays = []

# Number of rows to process
max_rows = 1

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    if index >= max_rows:
        break
    instruction = row["task"]

    essay = generate_essay(instruction)
    essays.append(essay)

    # Print progress
    print(f"Processed row {index + 1}/{len(df)}")

# Add the results to a new column in the dataframe
df["essay_result"] = essays + [None] * (len(df) - len(essays))

# Select only the "essay_result" and "instructions" columns
# result_df = df[['instructions', 'essay_result']]

# Save the updated dataframe to a new CSV file
df.to_csv("./daigt_updated.csv", index=False)
