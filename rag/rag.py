import dspy 


class GenerateAnswer(dspy.Signature):
    """custom signature function that can be defined """

    context=dspy.InputField(list, description="Retrieved context passages")
    question=dspy.InputField()
    answer=dspy.OutputField(str, description="Generated answer based on context and question")

class RAG(dspy.Module):

    def __init__(self, k=5):
        super().__init__()

        self.retrieve=dspy.Retrieve(k=4)
        self.gen_answer=dspy.ChainOfThoughts(" signature is added here, initial prompt instructuion")
        # if using custom signature use class GenerateAnswer
        self.gen_answer=dspy.ChainOfThoughts(GenerateAnswer)
    
    def forward(self, question):

        context=[]
        context=self.retrieve(question).passages
        prediction=self.gen_answer(question=question, context= context)

        return dspy.Prediction(context=context, answer=prediction.answer)