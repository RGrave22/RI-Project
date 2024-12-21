import numpy as np
import pandas as pd
import ReRankLMD
import requests
import OpenSearchSimpleAPI as osearch
import pprint as pp

def generate_text_with_ollama(prompt):
            """
            Sends a prompt to an Ollama server and retrieves the generated text.

            Args:
                prompt (str): The text prompt to send to the model.

            Returns:
                str: Generated text from the model.
            """
            endpoint = "https://twiz.novasearch.org/ollama"
            data = {
                "model": 'llama3.2',
                "stream": False,
                "prompt": prompt
            }
            print(data)

            try:
                response = requests.post(endpoint, json=data)
                response.raise_for_status()
                return response.json()["response"]
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                return None

class Summarization:

    def Summarize(self, topDocs, numDocs ):
        print("========================================SUMARIZATION BEGIN=====================================")

        opensearch = osearch.OSsimpleAPI()

        tmp = topDocs[:numDocs]['_source.doc']

        aux = []
        for doc in tmp:
            res = opensearch.get_doc_body(doc)
            print(res)
            print()
            aux.append(res)

        toSummarize = "[SEP]".join(aux)
        
        print("====================================SUMMARIZATION==================================================")

        print("BEFORE: " + toSummarize)

        #Prompt = "Summarize this documents and do not use more than 50 words (the docs are separated by [SEP]): "

        Prompt = "Sumarize the following docs that are separated by [Sep] in no more than 50 words You answer should only be the summarization nothing more: "

        PROMPT = Prompt + toSummarize

        response_text = generate_text_with_ollama(PROMPT)

        if response_text:
            print("===========================RESPONSE SUMARIZE=====================================================================")
            print(response_text)
            print("-------------------")
        else:
            print("Failed to retrieve a response.")


