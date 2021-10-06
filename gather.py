import os
import json
import time
import requests

HEADERS = {"Authorization": os.environ["TOKEN"]}

def gather_messages(author_id=791950104680071188, 
                    channel_id=870330763772563481,
                    offset=0):
        url = ("https://discord.com/api/v9/guilds/870330763772563476/messages/search?"
                f"author_id={author_id}&channel_id={channel_id}&offset={offset}")
    
        return requests.get(url, headers=HEADERS).json()

def gather_context(messages=gather_messages(), **kwargs):
    conversations = []
    if kwargs["limit"] != None and type(kwargs["limit"]) == int:
        messages = messages["messages"][:kwargs["limit"]]
    else:
        messages = messages["messages"]

    for idx, message in enumerate(messages):
        message = message[0] # Not sure why it is an array when there is always just one dict
        msg_id = message["id"]
        content = message["content"]
        context_url = f"https://discord.com/api/v9/channels/870330763772563481/messages?limit=3&around={msg_id}"

        resp = requests.get(context_url, headers=HEADERS).json()
        sorted_resp = sorted(resp, key=lambda i: i["id"])
        context_content = sorted_resp[0]["content"]
        conversations.append({"tag": idx, "context": context_content, "reply": content})
        print(f"Loaded {idx+1} conversation group")
        time.sleep(1) # Preventing rate limit

    if kwargs.get("save"):
        if os.path.isfile("conversations.txt"):
            inp = input("Found existing conversations.txt file, do you wish to override it? y\\n: ")
            if inp != "y":
                return
        file = open("conversations.txt", "w+")
        file.write(json.dumps(conversations))
        file.close()
        print("Saved gathered messages in messages.txt file")
    else:
        print("-"*100, f"\n{conversations}\n", "-"*100)

gather_context(limit=2)