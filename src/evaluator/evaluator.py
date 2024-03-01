
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
import logging
import json
import logging.config
import os
import glob
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, criteria, dataset_paths, model_name="gpt-3.5-turbo", logger_config=None):
        self.criteria = criteria
        self.dataset_paths = dataset_paths
            
        # Generate folder name with current date and time
        folder_name = f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        # Create the folder
        opath = Path(f"{self.dataset_paths['output_path']}/{folder_name}/")
        opath.mkdir(parents=True, exist_ok=True)
        self.dataset_paths['output_path']=opath
        self.model_name = model_name
        self.logger = self.setup_logger(logger_config)
        self.eval_chain = self.initialize_eval_chain()

    def setup_logger(self, logger_config):
        if logger_config:
            logging.config.dictConfig(logger_config)
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def initialize_eval_chain(self):
        # Initialize QAEvalChain with the specified language model
        return QAEvalChain.from_llm(llm=ChatOpenAI(model_name=self.model_name, temperature=0))

    def load_datasets(self):
        eval_df = pd.read_csv(self.dataset_paths['eval_questions'])
        llm_answers = pd.read_csv(self.dataset_paths['llm_answers'])
        return eval_df, llm_answers

    def grade_answers(self, predicted_dataset, predictions, criterion, prediction_key="result"):
        """
        Grades the answer based on ground truth and model predictions.
        @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
        @param predictions: A list of dictionaries containing model predictions for the questions.
        @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
        @param logger: logger
        @return: A list of scores for the distilled answers.
        """
        self.logger.info(f"Grading answers based on {criterion}")
        with_context = False
        logger.info("`Grading model answer ...`")
        if criterion == "correctness":
            prompt = GRADE_ANSWER_PROMPT
        elif criterion == "coherence":
            prompt = COHERENCE_PROMPT
        elif criterion == "contextuality":
            with_context = True
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
        if with_context:
            # Loop through each item in the dataset and evaluate it
            for item in predicted_dataset:
                # Ensure the item contains 'context', 'question', and the prediction key (e.g., 'result')
                context_file_path = item.get('context', '')

                # Assuming 'context' is a path to the file containing the context information
                # and you want to read this file's content into the 'answer' field
                try:
                    with open(context_file_path, 'r', encoding='utf-8') as file:
                        context_content = file.read()
                    # Update the 'answer' field with the content of the context file
                    item['answer'] = context_content
                except FileNotFoundError:
                    logger.error(f"File not found: {context_file_path}")
                    item['answer'] = "Error: Context file not found."
                        
            graded_outputs = eval_chain.evaluate(predicted_dataset,predictions, question_key="question", prediction_key=prediction_key)
        else:

            # print(predicted_dataset)
            graded_outputs = eval_chain.evaluate(predicted_dataset,
                                                predictions,
                                                question_key="question",
                                                prediction_key=prediction_key)
        logger.info(graded_outputs)
        return graded_outputs
    
    def save_graded_answers(self, questions, llm_answer, data, llm):
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

        # Function to extract grade
        def extract_grade(text):
            if pd.isnull(text):
                return None
            try:
                grade = text.split('\n')[0]  # Split by newline and get the first part
                grade_number = int(grade.split(': ')[1])  # Split by ': ' and convert the second part to integer
                return grade_number
            except (IndexError, ValueError):
                return None

        # Apply the function to each cell in the DataFrame
        df_grades_only = df.applymap(extract_grade)
        path = f"{self.dataset_paths['output_path']}/{llm}_evaluated_grade_only.csv"
        df_grades_only.to_csv(path,index=False)

        df['questions']=questions['question']
        df['correct_answer']=questions['answer']
        df[f'predicted_answer'] = llm_answer
        path = f"{self.dataset_paths['output_path']}/{llm}_evaluated.csv"
        df.to_csv(path,index=False)
        # Display the DataFrame
        print(df)
        return df


    def evaluate(self):
        eval_df, llm_answers = self.load_datasets()
        Path(self.dataset_paths['output_path']).mkdir(parents=True, exist_ok=True)
        
        for llm in llm_answers.columns:
            self.logger.info(f"Currently grading LLM: {llm}")
            results = {}
            for criterion in self.criteria:
                graded_answers = self.grade_answers(predicted_dataset=eval_df.to_dict('records'),
                                                    predictions=pd.DataFrame(llm_answers[llm]).to_dict('records'),
                                                    criterion=criterion,
                                                    prediction_key=llm)
                results[criterion] = graded_answers # Assuming each result is a dict with a 'score'
                
            
            path = f"{self.dataset_paths['output_path']}"
            logger.info(f"data saved to {path}")

            self.save_graded_answers(eval_df, llm_answers[llm].tolist(), results, llm)
            self.interpret()

    def interpret(self):
        directory = f"{self.dataset_paths['output_path']}/"
        # Pattern to match files ending with '_grade_only.csv'
        pattern = os.path.join(directory, '*_grade_only.csv')

        # Placeholder for aggregated data
        aggregated_data = []

        # Loop through all files matching the pattern
        for file_path in glob.glob(pattern):
            # Extract LLM name from the filename
            llm_name = os.path.basename(file_path).split('_evaluated_grade_only.csv')[0]
            df = pd.read_csv(file_path)

            column_means = df.mean()
            column_means['mean'] = column_means.mean(axis=1)  # Adjust column indexing as necessary if your data starts from a different column
        
            column_means['LLM'] = llm_name
            aggregated_data.append(column_means)
            
   
        # Combine all aggregated data into a new DataFrame
        combined_df = pd.DataFrame(aggregated_data)
        # Save the aggregated DataFrame to a new file
        combined_df.to_csv(os.path.join(directory, 'aggregated_evaluation.csv'),index=False)

if __name__ == "__main__":
    # logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    # Load logging configuration from a JSON file
    with open('logging_config.json', 'r') as f:
        config = json.load(f)
        logging.config.dictConfig(config)
    
    parser = argparse.ArgumentParser(description='Grade LLM answers based on multiple criteria.')
    parser.add_argument('--criteria', nargs='+', help='List of grading criteria', default=['correctness','coherence', 'contextuality', 'informativeness', 'fluency'])
    parser.add_argument('--output', help='Output path to store evaluation data', default='./data/eval_output/')
    parser.add_argument('--question', help='path for evaluation question dataset', default='data/eval_input/eval_questions.csv')
    parser.add_argument('--evaluation',  help='path for LLM result to be evaluated', default='data/eval_input/llm_answers.csv')

    args = parser.parse_args()
    if len(args.criteria) == 1:
        criterion = args.criteria[0].split('+')
    else:
        criterion = args.criteria

    dataset_paths = {
        'eval_questions': args.question,
        'llm_answers': args.evaluation,
        'output_path': args.output
    }

    logging.info("Paths passed in through args: {}".format(dataset_paths))
    logging.info("Criterion to be evaluated on: %s", criterion)

    evaluator = ModelEvaluator(criterion, dataset_paths,logger_config=config)
    evaluator.evaluate()


