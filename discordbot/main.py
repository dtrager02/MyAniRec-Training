import discord
from discord import app_commands

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        print(f'Message from {message.author}: {message.content}')

intents = discord.Intents.default()
intents.message_content = True
TOKEN = open('token.txt',"r").read()
client = MyClient(intents=intents)
client.run('my token goes here')