import os
import json
from time import sleep
import requests

HEADERS = {"Authorization": os.environ["TOKEN"]}

def gather_messages(file_name="messages", 
                    author_id=791950104680071188, 
                    channel_id=870330763772563481):

    if not os.path.isfile(f"{file_name}.txt"):
        url = ("https://discord.com/api/v9/guilds/870330763772563476/messages/search?"
                f"author_id={author_id}&channel_id={channel_id}&offset=0")
        messages = requests.get(url, headers=HEADERS).json()
        file = open(f"{file_name}.txt","w+")
        file.write(json.dumps(messages))
        file.close()
        print("Gathered messages.")


f = json.loads(open("messages.txt", "r").read())

for message in f["messages"]:
    message = message[0]
    msg_id = message["id"]
    content = message["content"]
    print(f"Content: {content}")

    context = f"https://discord.com/api/v9/channels/870330763772563481/messages?limit=3&around={msg_id}"
    resp = requests.get(context, headers=HEADERS).json()
    sorted_resp = sorted(resp, key=lambda i: i["id"])

    print(sorted_resp[0]["content"])
    break

