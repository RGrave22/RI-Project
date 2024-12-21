import pandas as pd
import ReRankLMD
import requests

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

class rank_with_history:

    def rankWithHistoryLlama(self, first_utterance, turn, opensearch):
        rr = ReRankLMD.ReRank()   
        query = first_utterance + " " + turn['raw_utterance']

        print("========================================QUERY REWRITE==============================================")
        print("QUERY BEFORE: " + query)

        #Prompt = "Rewrite the current utterance (the last question) using past utterances (the previous questions), YOUR answer should only be the rewritten query, nothing more. Query to be rewritten: "

        #Prompt = "Rewrite the current sequence of questions giving a new question that is a combination of the previous questions. The new question should be a question that is likely to be asked next. Your answer should only be the rewritten query, nothing more. The previous questions are:"

        Prompt = "Rewrite this prompt in a precise and concise manner that contains all the essential information. Your answer should only be the rewritten query, nothing more. Prompt: "


        PROMPT = Prompt + query

        response_text = generate_text_with_ollama(PROMPT)

        if response_text:
            print("=================================================QUERY AFTER REWRITE==============================================")
            print()
            print(response_text)
            print()
            print("==========================================================================================================")
        else:
            print("Failed to retrieve a response.")
        
        #LMD
        res = rr.reRank(topic = response_text, opensearch = opensearch, numDocs=50)

        #print("\nQUERY: " + query + "\n")

        return res

    # def rankWithHistory_title(self, topic, turn, opensearch):

    #     query = topic['title'] + " " + turn['raw_utterance']

    #     res = opensearch.search_body(query, numDocs=100)

    #     return res

    

