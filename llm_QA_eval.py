
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import MosaicML
from langchain_community.llms import Anthropic
from langchain_community.llms import Replicate
import csv
import logging
from mlflow.metrics.genai import relevance, EvaluationExample
import argparse
from metrics import *
# relevance_metric = relevance(model="openai:/gpt-4")
# print(relevance_metric)
import pandas as pd
from datetime import datetime
from pathlib import Path

def grade_model_answer(predicted_dataset, predictions, criterion, logger,prediction_key = "result"):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @param logger: logger
    @return: A list of scores for the distilled answers.
    """

    logger.info("`Grading model answer ...`")
    if criterion == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif criterion == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif criterion == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    elif criterion == "coherence":
        prompt = COHERENCE_PROMPT
    elif criterion == "contextuality":
        prompt = CONTEXTUALITY_PROMPT
    elif criterion == "informativeness":
        prompt = INFORMATIVENESS_PROMPT
    elif criterion == "fluency":
        prompt = FLUENCY_PROMPT
    else:
        prompt = GRADE_ANSWER_PROMPT


    # Note: GPT-4 grader is advised by OAI 
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                                      prompt=prompt)
    # print(predicted_dataset)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key=prediction_key)
    return graded_outputs

def load_dataset():
    # with open('eval_question.csv', mode='r') as file:
    #     reader = csv.DictReader(file)
    #     eval_df = [row for row in reader]
    #     # print(eval_df)
    # # eval_df = pd.read_csv('eval_question.csv')
        

    # # Path to your CSV file
    # csv_file_path = 'result.csv'

    # # Read the CSV file
    # # {llm1: [{result: prediction1},{result: prediction2},(result:prediction3)...]}
    # with open(csv_file_path, mode='r', newline='') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     llm_answers = {llm: [] for llm in reader.fieldnames}
    #     # Collect answers for each LLM
    #     for row in reader:
    #         for llm in reader.fieldnames:
    #             llm_answers[llm].append({"result": row[llm]})
    eval_df = pd.read_csv('eval_question.csv')
    llm_answers = pd.read_csv('graded_answers.csv')

    return eval_df, llm_answers
def save_graded_answers(question,llm_answer,data,name):
    # Initialize an empty list to store the row data
    rows = []

    # Calculate the maximum length among the values in the data dictionary
    max_length = max(len(value) for value in data.values())

    # Loop through each index up to the maximum length
    for i in range(max_length):
        # Create a dictionary for the current row
        row = {}
        # Loop through each key in the data dictionary
        for key, value in data.items():
            # Check if the current index is within the range of the current list
            if i < len(value):
                # Add the result to the row dictionary
                row[key] = value[i]['results']
            else:
                # Add a None value if the index is out of range
                row[key] = None
        # Add the row dictionary to the list of rows
        rows.append(row)

    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows)
    df['questions']=question['question']
    df['correct_answer']=question['answer']
    df[f'predicted_answer'] = llm_answer
    df.to_csv(name)
    # Display the DataFrame
    print(df)
    return df

def main():
    # logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    # Generate folder name with current date and time
    folder_name = f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Create the folder
    path = Path(f"./data/{folder_name}")
    path.mkdir(parents=True, exist_ok=True)
    
    
    parser = argparse.ArgumentParser(description='Grade LLM answers based on multiple criteria.')
    parser.add_argument('--criteria', nargs='+', help='List of grading criteria', default=['fast','coherence', 'contextuality', 'informativeness', 'fluency'])
    args = parser.parse_args()

    criterion = args.criteria[0].split('+')
    eval_df, llm_answers = load_dataset()
    # Loop through each LLM and grade their answers

# ---------------GENERATE EVALUATION--------------------
    
 # Iterate over each LLM
    for llm in llm_answers.columns:

        print(llm,criterion)

        # Results dictionary for this LLM
        results = {}
        # # Iterate over selected criteria
        for criteria in criterion:
            graded_answers = grade_model_answer(predicted_dataset=eval_df.to_dict('records'), predictions=pd.DataFrame(llm_answers[llm]).to_dict('records'),prediction_key=llm, criterion=criteria, logger=logger)
            results[criteria] = graded_answers
        # Now results contain graded answers for all selected criteria
        # Process or save these results as needed
        csv_name = f"{path}/{llm}_eval.csv"
        save_graded_answers(eval_df,llm_answers[llm],results,csv_name)


if __name__=="__main__":
    main()
