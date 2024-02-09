



eval_df = [
    {"question": "What is MLflow?",
    "answer": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle, including experiment tracking, model management, and serving."
},{"question":"How does useEffect() work?",
    "answer":"The useEffect() hook tells React that your component needs to do something after render. React will remember the function you passed (we’ll refer to it as our “effect”), and call it later after performing the DOM updates."}]

headers = ['question', 'answer']
with open('data.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    for item in eval_df:
        writer.writerow(item)
