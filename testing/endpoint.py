import requests

url = "http://51.159.138.187:8000/summerize"


data = {
    "query": "what is my name", 
    "documents":[{
        "content":"my name is lennart and I am 20 years old", 
        "metadata":{"file_name": "me.txt"}
        }
    ]
}

r = requests.post(url, data)