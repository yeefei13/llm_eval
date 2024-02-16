    
import csv

import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline


# Load tokenizer and model
model_name = "facebook/bart-large-cnn"  # This model is fine-tuned for summarization tasks
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize the text generation pipeline with BART
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="pt")

# Load questions from CSV
questions_df = pd.read_csv("eval_question.csv")

# Generate answers
answers = []
for question in questions_df['question']:
    # Generate an answer
    inputs = tokenizer.encode("Question: " + question + " Answer:", return_tensors="pt", add_special_tokens=True)
    summary_ids = model.generate(inputs, max_length=50, min_length=10, length_penalty=3.0, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary_ids)
    
    # Extract the answer from the generated text
    answer = answer.replace("Question: " + question + " Answer:", "").strip()
    answers.append(answer)

# Append the answers as a new column to the DataFrame
questions_df['BART'] = answers

# Save the updated DataFrame back to CSV
questions_df.to_csv("result.csv", index=False)

print("Answering complete. Answers appended and saved to result.csv.")



# generated_answer = [{"result": "an opensource framework that helps making llm pipeline"}, 
#                     {"result": "a react function that defines how to react after a render(you can pass in a function)"}]   
# headers = ['result']
# with open('result.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=headers)
#     writer.writeheader()
#     for item in generated_answer:
#         writer.writerow(item)