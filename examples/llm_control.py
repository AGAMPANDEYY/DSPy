import dspy

class MultiHopQA(dspy.Module):

    def __init__(self):
        self.retrieve= dspy.Retrieve(k=3) 
        self.gen_query=dspy.ChainOfThought("")
        self.gen_answer=dspy.ChainOfThought("")
    
    def forward(self, question):

        context=[]

        for hop in range(2):

            query= self.gen_query(f"Generate a search query for the question: {question} given the context: {context}").query
            context=self.retrieve(query).passages

            return self.gen_answer(f"Using the context: {context}, answer the question: {question}")
