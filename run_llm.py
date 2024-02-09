    
import csv




generated_answer = [{"result": "an opensource framework that helps making llm pipeline"}, 
                    {"result": "a react function that defines how to react after a render(you can pass in a function)"}]   
headers = ['result']
with open('result.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    for item in generated_answer:
        writer.writerow(item)