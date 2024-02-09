
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import MosaicML
from langchain_community.llms import Anthropic
from langchain_community.llms import Replicate
import csv
import logging
from mlflow.metrics.genai import relevance, EvaluationExample


# relevance_metric = relevance(model="openai:/gpt-4")
# print(relevance_metric)


template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:

Your response should be as follows:

GRADE: (Correct or Incorrect)
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect. Use one or two sentences maximum. Keep the answer as concise as possible.)
"""

GRADE_ANSWER_PROMPT = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""

GRADE_ANSWER_PROMPT_FAST = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.
You are also asked to identify potential sources of bias in the question and in the true answer.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:

Your response should be as follows:

GRADE: (Correct or Incorrect)
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect, identify potential sources of bias in the QUESTION, and identify potential sources of bias in the TRUE ANSWER. Use one or two sentences maximum. Keep the answer as concise as possible.)
"""

GRADE_ANSWER_PROMPT_BIAS_CHECK = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

template = """You are assessing a submitted student answer to a question relative to the true answer based on the provided criteria: 
    
    ***
    QUESTION: {query}
    ***
    STUDENT ANSWER: {result}
    ***
    TRUE ANSWER: {answer}
    ***
    Criteria: 
      relevance:  Is the submission referring to a real quote from the text?"
      conciseness:  Is the answer concise and to the point?"
      correct: Is the answer correct?"
    ***
    Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print "Correct" or "Incorrect" (without quotes or punctuation) on its own line corresponding to the correct answer.
    Reasoning:
"""

GRADE_ANSWER_PROMPT_OPENAI = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

template = """ 
    Given the question: \n
    {query}
    Here are some documents retrieved in response to the question: \n
    {result}
    And here is the answer to the question: \n 
    {answer}
    Criteria: 
      relevance: Are the retrieved documents relevant to the question and do they support the answer?"
    Do the retrieved documents meet the criterion? Print "Correct" (without quotes or punctuation) if the retrieved context are relevant or "Incorrect" if not (without quotes or punctuation) on its own line. """


def grade_model_answer(predicted_dataset, predictions, grade_answer_prompt, logger,prediction_key = "result"):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @param logger: logger
    @return: A list of scores for the distilled answers.
    """

    logger.info("`Grading model answer ...`")
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Note: GPT-4 grader is advised by OAI 
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key=prediction_key)
    return graded_outputs


def main():
    # logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    with open('eval_question.csv', mode='r') as file:
        reader = csv.DictReader(file)
        eval_df = [row for row in reader]
        # print(eval_df)
    

    # Path to your CSV file
    csv_file_path = 'result.csv'

    # Read the CSV file
    with open(csv_file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        llm_answers = {llm: [] for llm in reader.fieldnames}
        # Collect answers for each LLM
        for row in reader:
            for llm in reader.fieldnames:
                llm_answers[llm].append({"result": row[llm]})
    # Loop through each LLM and grade their answers
    print(llm_answers)
# ---------------GENERATE EVALUATION--------------------
    
    # Prepare a dictionary to hold all graded answers, keyed by LLM
    graded_answers_dict = {}
    
    # Iterate over each LLM and its answers
    for llm, answers in llm_answers.items():
        # Grade the answers
        graded_answers = grade_model_answer(predicted_dataset=eval_df, predictions=answers, grade_answer_prompt="Descriptive w/ bias check", logger=logger)
        
        # Store the graded answers for this LLM
        graded_answers_dict[llm] = graded_answers
    

    # for testing
    # graded_answers_dict = {'llm1': [{'results': "GRADE: Incorrect\n\nJUSTIFICATION: The student's answer is incorrect because it does not accurately describe MLflow as a platform for managing the end-to-end machine learning lifecycle. The question and true answer do not appear to contain any bias."}, {'results': "GRADE: Correct\n\nJUSTIFICATION: The student's answer accurately describes the function of useEffect() in React, although it is less detailed than the true answer. There is no apparent bias in the question or the true answer."}], 'llm2': [{'results': "GRADE: Correct\n\nJUSTIFICATION: The student's answer correctly identifies MLflow as a platform for managing end-to-end machine learning projects, although it lacks the detail of the true answer. There's no apparent bias in the question or the true answer."}, {'results': "GRADE: Correct\n\nJUSTIFICATION: The student's answer is correct as it accurately describes the basic functionality of useEffect() in React. There is no apparent bias in the question or the true answer."}]}


    # Prepare CSV file headers (LLM names) and rows
    headers = list(graded_answers_dict.keys())
    num_questions = len(eval_df)  # Assuming each question receives a single graded answer


    # Writing to CSV
    with open('graded_answers.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        writer.writeheader()
        
        # For each question, write a row with each LLM's graded answer for that question
        for i in range(num_questions):
            row = {llm: graded_answers_dict[llm][i] for llm in headers}
            writer.writerow(row)

if __name__=="__main__":
    main()
