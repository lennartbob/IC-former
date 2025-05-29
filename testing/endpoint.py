import requests

url = "http://51.159.138.187:8000/summarize/" # Corrected spelling


data = {
    "query": "what is my name",
    "document":{ 
        "content":"my name is lennart and I am 20 years old",
        "metadata":{"file_name": "me.txt"}
    },
    "expaned_queries":None
}
# Send JSON data

r = requests.post(url, json=data)

print(r.status_code)
print(r.json())