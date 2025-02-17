from agno.workflow import Workflow, RunResponse

class MyFirstWorkflow(Workflow):
    def run(self):
        yield RunResponse(content='Hello from My Workflow!')

if __name__ == "__main__":
    workflow = MyFirstWorkflow()
    for response in workflow.run():
        print(response.content)