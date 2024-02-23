from langchain.prompts import PromptTemplate

# Revised GRADE_ANSWER_PROMPT with a 1-5 scale and structured response format.
GRADE_ANSWER_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"],
    template="""Given a question, evaluate the student's answer compared to the true answer on a scale from 1 to 5 for factual accuracy. Provide a grade followed by a brief explanation for your grading decision.

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

give me the grade(1-5) and explaination in this format:
Grade:
Explanation:
"""
)

# Revised COHERENCE_PROMPT with structured response format.
COHERENCE_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"],
    template="""Assess the coherence of the student's answer relative to the question and true answer. Grade on a scale from 1 to 5, then provide an explanation for your grade.

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

give me the grade(1-5) and explaination in this format:
Grade:
Explanation:
"""
)

# Revised CONTEXTUALITY_PROMPT with structured response format.
CONTEXTUALITY_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"],
    template="""Evaluate the relevance and integration of the student's answer with the given context. Use a 1 to 5 scale for your grade and follow with an explanation.

CONTEXT: {result}
QUESTION: {query}
MODEL ANSWER: {answer}

give me the grade(1-5) and explaination in this format, in the explaination, quote the sentences in the context that the model answer corresponse to:
Grade:
Explanation:
"""
)

# Revised INFORMATIVENESS_PROMPT with structured response format.
INFORMATIVENESS_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"],
    template="""Determine how informative the student's answer is, using a scale from 1 to 5. Provide both the grade and an explanation for your evaluation.

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

give me the grade(1-5) and explaination in this format:
Grade:
Explanation:
"""
)

# Revised FLUENCY_PROMPT with structured response format.
FLUENCY_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"],
    template="""Evaluate the fluency of the student's answer on a scale from 1 to 5. After grading, explain the reasoning behind your evaluation.

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

give me the grade(1-5) and explaination in this format:
Grade:
Explanation:
"""
)

# These adjustments ensure that the grading and explanation are clearly delineated, improving the readability and consistency of the evaluations.
